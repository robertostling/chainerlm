import sys
import pickle
import argparse
import random
import time

import chainer
from chainer import cuda, Variable
from chainer import optimizers, serializers
import chainer.functions as F

from lm import LanguageModel
from zipcorpus import ZipCorpus

def main():
    parser = argparse.ArgumentParser(
            description='LSTM language model training')
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
            '--chunk-size', type=int, metavar='N', default=1024,
            help='size of text chunks')
    parser.add_argument(
            '--layers', type=int, metavar='N', default=2,
            help='number of LSTM layers')
    parser.add_argument(
            '--batch-size', type=int, metavar='N', default=64,
            help='batch size')
    parser.add_argument(
            '--dropout', type=float, metavar='X', default=0.0,
            help='dropout factor')
    parser.add_argument(
            '--vocabulary', type=str, metavar='FILE',
            help='pickled character vocabulary file')
    parser.add_argument(
            '--corpus', type=str, metavar='FILE',
            help='zip file containing raw text documents')
    args = parser.parse_args()

    if args.vocabulary:
        with open(args.vocabulary, 'rb') as f:
            vocab = pickle.load(f)
    else:
        vocab = list(
                r''' !"&\'()*+,-./0123456789:;<>?ABCDEFGHIJKLMNOPQRS'''
                r'''TUVWXYZ_abcdefghijklmnopqrstuvwxyzÄÅÖäåö–“”…''')


    vocab = ['<UNK>'] + vocab
    vocab_index = {c:i for i,c in enumerate(vocab)}

    model = LanguageModel(
            vocab, args.embedding_size, args.lstm_size, args.layers,
            dropout=args.dropout)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    xp = model.xp

    zipcorpus = ZipCorpus(args.corpus)

    streams = [zipcorpus.character_stream(args.chunk_size)
               for _ in range(args.batch_size)]

    def read_batch():
        rows = [next(stream) for stream in streams]
        unk = vocab_index['<UNK>']
        return xp.array([
            [vocab_index.get(c, unk) for c in row]
            for row in rows],
            dtype=xp.int32)

    n_heldout_batches = 8
    heldout_batches = [read_batch() for _ in range(n_heldout_batches)]

    optimizer = optimizers.Adam()
    optimizer.use_cleargrads()
    optimizer.setup(model)

    train_hs, train_cs = None, None
    while True:
        t0 = time.time()
        batch = read_batch()
        pred_ys, train_hs, train_cs = model(batch, train_hs, train_cs)
        pred_ys = F.stack(pred_ys, axis=0)
        loss = F.softmax_cross_entropy(
                F.reshape(
                    pred_ys[:,:-1],
                    (pred_ys.shape[0]*(pred_ys.shape[1]-1), pred_ys.shape[2])),
                F.flatten(batch[:,1:]))
        model.cleargrads()
        loss.backward()
        optimizer.update()
        print('TRAIN', cuda.to_cpu(loss.data), time.time()-t0, flush=True)


if __name__ == '__main__':
    with chainer.using_config('use_cudnn', 'never'):
        main()

