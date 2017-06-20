import sys
import pickle
import argparse

import numpy as np

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

from lm import LanguageModel

def main():
    parser = argparse.ArgumentParser(
            description='LSTM language model text annotation tool')
    parser.add_argument(
            '--normalize-space', action='store_true',
            help='normalize blank spaces before scoring')
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
            '--model', type=str, metavar='PREFIX',
            help='prefix of model files')
    parser.add_argument(
            'filenames', type=str, metavar='FILE', nargs='+')
    args = parser.parse_args()

    with open(args.model + '.metadata', 'rb') as f:
        train_args = pickle.load(f)
        vocab = pickle.load(f)

    vocab_index = {c:i for i,c in enumerate(vocab)}

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
            symbols = [vocab_index[c] for c in text if c in vocab_index]
        else:
            unk = vocab_index['<UNK>']
            symbols = [vocab_index.get(c, unk) for c in text]

        for i in range(0, len(symbols), args.chunk_size-1):
            chunk = symbols[i:i+args.chunk_size]
            yield xp.array([chunk], dtype=xp.int32)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    for filename in args.filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        if args.normalize_space:
            text = ' '.join(text.split())

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

