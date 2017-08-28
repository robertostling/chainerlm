import sys
import pickle
import argparse
import random
import time
import math

import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
import chainer.functions as F

from lm import MultiLanguageModel
#from zipcorpus import ZipCorpus

def main():
    parser = argparse.ArgumentParser(
            description='Multilingual LSTM language model training')
    parser.add_argument(
            '--gpu', type=int, metavar='N', default=-1,
            help='gpu to use (default: use CPU)')
    parser.add_argument(
            '--lstm-size', type=int, metavar='N', default=256,
            help='number of LSTM units')
    parser.add_argument(
            '--embedding-size', type=int, metavar='N', default=32,
            help='dimensionality of character embeddings')
    parser.add_argument(
            '--language-embedding-size', type=int, metavar='N', default=64,
            help='dimensionality of language embeddings')
    parser.add_argument(
            '--chunk-size', type=int, metavar='N', default=1024,
            help='size of text chunks')
    parser.add_argument(
            '--layers', type=int, metavar='N', default=2,
            help='number of LSTM layers')
    parser.add_argument(
            '--random-seed', type=int, metavar='N',
            help='random seed for data shuffling (and heldout set sampling)')
    parser.add_argument(
            '--batch-size', type=int, metavar='N', default=64,
            help='batch size')
    parser.add_argument(
            '--dropout', type=float, metavar='X', default=0.0,
            help='dropout factor')
    parser.add_argument(
            '--vocabulary', type=str, metavar='FILE', required=True,
            help='pickled character vocabulary file')
    parser.add_argument(
            '--corpus', type=str, metavar='FILE',
            help='UTF-8 encoded file in language<tab>text format')
    parser.add_argument(
            '--model', type=str, metavar='PREFIX',
            help='prefix of model files')
    args = parser.parse_args()

    with open(args.vocabulary, 'rb') as f:
        vocab = pickle.load(f)

    vocab = ['<UNK>'] + vocab
    vocab_index = {c:i for i,c in enumerate(vocab)}

    with open(args.corpus, 'r', encoding='utf-8') as f:
        data = [[field.strip() for field in line.split('\t')]
                for line in f]
        data = [fields for fields in data if len(fields) == 2]

    languages = sorted({fields[0] for fields in data})
    random.shuffle(data)

    model = MultiLanguageModel(
            vocab, len(languages), args.language_embedding_size,
            args.embedding_size, args.lstm_size, args.layers,
            dropout=args.dropout)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    xp = model.xp

    if args.random_seed: random.seed(args.random_seed)
    #zipcorpus = ZipCorpus(args.corpus)
    #streams = [zipcorpus.character_stream(args.chunk_size)
    #           for _ in range(args.batch_size)]

    #def read_batch():
    #    rows = [next(stream) for stream in streams]
    #    unk = vocab_index['<UNK>']
    #    return xp.array([
    #        [vocab_index.get(c, unk) for c in row]
    #        for row in rows],
    #        dtype=xp.int32)

    def text_stream(language):
        i = 0
        buf = ''
        while True:
            if data[i][0] == language:
                sentence = data[i][1]
                if buf:
                    buf = buf + ' ' + sentence
                else:
                    buf = sentence
                while len(buf) >= args.chunk_size:
                    yield buf[:args.chunk_size]
                    buf = buf[args.chunk_size:]
            else:
                pass
            i = (i + 1) % len(data)

    streams = [text_stream(language) for language in languages]

    def read_batch(batch_size):
        unk = vocab_index['<UNK>']
        langs = [random.randint(0, len(languages)-1)
                 for _ in range(batch_size)]
        xs = [next(streams[lang]) for lang in langs]
        return (xp.array(langs, dtype=xp.int32), 
                xp.array([[vocab_index.get(c, unk) for c in x] for x in xs],
                         dtype=xp.int32))

    n_heldout_batches = 8
    heldout_batches = [read_batch(args.batch_size)
                       for _ in range(n_heldout_batches)]

    optimizer = optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5.0))

    def compute_loss(langs, xs, hs, cs):
        pred_ys, hs, cs = model(langs, xs, hs, cs)
        pred_ys = F.stack(pred_ys, axis=0)
        loss = F.softmax_cross_entropy(
                F.reshape(
                    pred_ys[:,:-1],
                    (pred_ys.shape[0]*(pred_ys.shape[1]-1), pred_ys.shape[2])),
                F.flatten(xs[:,1:]))
        return loss, pred_ys, hs, cs
 
    best_heldout_loss = float('inf')
    n_batches = 0
    #train_hs, train_cs = None, None
    while True:
        # NOTE: this allows the languages to vary between batches
        train_hs, train_cs = None, None
        t0 = time.time()
        langs, xs = read_batch(args.batch_size)
        model.cleargrads()
        loss, _, train_hs, train_cs = compute_loss(
                langs, xs, train_hs, train_cs)
        loss.backward()
        # NOTE: this is not needed when resetting train_hs, train_cs
        #loss.unchain_backward()
        optimizer.update()
        print('TRAIN',
              n_batches,
              cuda.to_cpu(loss.data)/math.log(2),
              time.time()-t0,
              flush=True)

        n_batches += 1
        if n_batches % 250 == 0:
            heldout_loss = 0.0
            t0 = time.time()
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    hs, cs = None, None
                    for langs, xs in heldout_batches:
                        loss, _, hs, cs = compute_loss(langs, xs, hs, cs)
                        heldout_loss += cuda.to_cpu(loss.data)
            print('DEV',
                  n_batches,
                  heldout_loss/(math.log(2)*len(heldout_batches)),
                  time.time()-t0,
                  flush=True)
            if heldout_loss < best_heldout_loss:
                best_heldout_loss = heldout_loss
                serializers.save_npz(args.model + '.npz', model)
                with open(args.model + '.metadata', 'wb') as f:
                    pickle.dump(args, f, -1)
                    pickle.dump(vocab, f, -1)


if __name__ == '__main__':
    with chainer.using_config('use_cudnn', 'auto'):
        main()

