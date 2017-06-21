import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

from lm import LanguageModel

class DecoderState:
    def __init__(self, model, hs=None, cs=None, history=()):
        self.model = model
        self.hs = hs
        self.cs = cs
        self.history = history

    def __call__(self, x_tm1, symbol=None):
        pred, hs, cs = self.model(x_tm1, self.hs, self.cs)
        state = DecoderState(
                self.model, hs=hs, cs=cs,
                history=self.history+(symbol,))
        state.pred = pred
        return state


class Hypothesis:
    def __init__(self, state, x_t,
                 log_p=0.0, distance=0.0, n_copied=0,
                 source_pos=0, edits=()):
        self.state = state
        self.x_t = x_t
        self.log_p = log_p
        self.distance = distance
        self.n_copied = n_copied
        self.source_pos = source_pos
        self.edits = edits
        self.score = log_p - 4*distance


class EditOp:
    def __init__(self):
        self.identity = False

class Insert(EditOp):
    def __init__(self, symbol, position):
        super().__init__()
        self.symbol = symbol
        self.position = position
        self.distance = 1.0

    def __str__(self):
        return '@%d+%d' % (self.position, self.symbol)

class SkipReplace(EditOp):
    def __init__(self, original, symbol, left, right, position):
        super().__init__()
        self.original = original
        self.symbol = symbol
        self.left = left
        self.right = right
        self.position = position
        self.identity = (original == symbol) and (left + right == 0)
        self.distance = float(original != symbol) + left + right

    def __str__(self):
        return '@%d %d,(%d->%d),%d' % (
                self.position, self.left, self.original, self.symbol,
                self.right)

def correct(model, text, beam_size=10, drop_unknown=True):
    xp = model.xp
    unk = model.alphabet_index['<UNK>']
    bos = model.alphabet_index[' ']

    if drop_unknown:
        text = [model.alphabet_index[c] for c in text
                if c in model.alphabet_index]
    else:
        text = [model.alphabet_index.get(c, unk) for c in text]

    beam = [Hypothesis(DecoderState(model), bos)]

    while True:
        new_states = {}
        best_at_pos = {}
        finished = []

        for hyp in beam:
            # Do not consider finished strings
            if hyp.source_pos == len(text):
                finished.append(hyp)
                continue

            new_state_id = (id(hyp.state), hyp.x_t)
            if new_state_id in new_states:
                state = new_states[new_state_id]
            else:
                state = hyp.state(
                        xp.array([[hyp.x_t]], dtype=xp.int32), hyp.x_t)
                state.log_pred = np.log2(
                        cuda.to_cpu(F.softmax(state.pred[0]).data)[0])
                new_states[new_state_id] = state

            for x_t, log_p in enumerate(state.log_pred):
                edits = [Insert(symbol=x_t, position=hyp.source_pos)]
                for skip_left, skip_right in [(0,0), (1,0), (0,1)]:
                    if hyp.source_pos+1+skip_left+skip_right > len(text):
                        continue
                    edits.append(SkipReplace(
                        original=text[skip_left+hyp.source_pos],
                        symbol=x_t,
                        left=skip_left, right=skip_right,
                        position=hyp.source_pos))
                for edit in edits:
                    if isinstance(edit, SkipReplace):
                        deleted = float(edit.left + edit.right)
                        distance = float(edit.symbol != edit.original) + deleted
                        pos = hyp.source_pos + 1 + edit.left + edit.right
                    elif isinstance(edit, Insert):
                        distance = 1.0
                        pos = hyp.source_pos

                    distance = edit.distance
                    #if hyp.n_copied > 0: distance *= 0.5

                    new_hyp = Hypothesis(
                            state, x_t,
                            log_p=hyp.log_p+log_p,
                            distance=hyp.distance+distance,
                            n_copied=0 if distance else hyp.n_copied+1,
                            source_pos=pos,
                            edits=hyp.edits+(edit,))
                    ident = (id(state), x_t, pos)
                    _, best_score = best_at_pos.get(
                            ident, (None, new_hyp.score))
                    if new_hyp.score >= best_score:
                        best_at_pos[ident] = (new_hyp, new_hyp.score)

        beam = sorted([hyp for hyp,_ in best_at_pos.values()] + finished,
                      key=lambda hyp: -hyp.score)[:beam_size]

        for hyp in beam:
            print(''.join(model.alphabet[c] for c in hyp.state.history) +
                  '|' + model.alphabet[hyp.x_t], hyp.log_p, hyp.distance)
        print('-'*72)

        # TODO: better criterion for termination?
        if beam[0].source_pos == len(text):
            break

    return beam

def main():
    parser = argparse.ArgumentParser(
            description='LSTM language model text annotation tool')
    parser.add_argument(
            '--normalize-space', action='store_true',
            help='normalize blank spaces before scoring')
    parser.add_argument(
            '--normalize', action='store_true',
            help='attempt to normalize a text with the language model')
    parser.add_argument(
            '--drop-unknown', action='store_true',
            help='drop unknown characters (default: use UNK tokens)')
    parser.add_argument(
            '--gpu', type=int, metavar='N', default=-1,
            help='gpu to use (default: use CPU)')
    parser.add_argument(
            '--chunk-size', type=int, metavar='N', default=1024,
            help='size of text chunks')
    parser.add_argument(
            '--beam-size', type=int, metavar='N', default=10,
            help='beam size for edit distance search')
    parser.add_argument(
            '--model', type=str, metavar='PREFIX',
            help='prefix of model files')
    parser.add_argument(
            'filenames', type=str, metavar='FILE', nargs='+')
    args = parser.parse_args()

    with open(args.model + '.metadata', 'rb') as f:
        train_args = pickle.load(f)
        vocab = pickle.load(f)

    model = LanguageModel(
            vocab, train_args.embedding_size, train_args.lstm_size,
            train_args.layers)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    xp = model.xp

    serializers.load_npz(args.model + '.npz', model)

    # Yield a sequence of overlapping (by one symbolsequences
    def get_batches(text):
        if args.drop_unknown:
            symbols = [model.alphabet_index[c] for c in text
                       if c in model.alphabet_index]
        else:
            unk = model.alphabet_index['<UNK>']
            symbols = [model.alphabet_index.get(c, unk) for c in text]

        for i in range(0, len(symbols), args.chunk_size-1):
            chunk = symbols[i:i+args.chunk_size]
            yield xp.array([chunk], dtype=xp.int32)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    from pprint import pprint
    for filename in args.filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        if args.normalize_space:
            text = ' '.join(text.split())

        if args.normalize:
            hypotheses = correct(
                    model, text,
                    beam_size=args.beam_size,
                    drop_unknown=args.drop_unknown)
            for hyp in hypotheses:
                print(hyp.source_pos,
                        ''.join(model.alphabet[c]
                                for c in hyp.state.history[1:] + (hyp.x_t,)))
                #print(hyp.source_pos)
                pprint([(model.alphabet[e.symbol], str(e)) for e in hyp.edits
                        if not e.identity])
        else:
            # Need a symbol to start predicting from
            text = ' ' + text

            hs, cs = None, None
            for batch in get_batches(text):
                pred, hs, cs = model(batch, hs, cs)
                pred = F.stack(pred, axis=0)
                xent = F.softmax_cross_entropy(
                        pred[0, :-1, :], batch[0, 1:],
                        reduce='no')
                pred = cuda.to_cpu(F.softmax(pred[0]).data)[:-1, :]
                xent = cuda.to_cpu(xent.data) / np.log(2.0)
                ent = -(pred*np.log2(pred)).sum(axis=-1)
                #print(pred.shape, xent.shape, ent.shape)
                #print(pred.sum(axis=-1))
                for c, c_xent, c_ent in zip(batch[0, 1:], xent, ent):
                    print('%s %.3f %.3f %.3f' % 
                            (vocab[c], c_xent, c_ent, c_xent-c_ent))


if __name__ == '__main__': main()
