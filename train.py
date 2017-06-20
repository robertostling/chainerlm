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
    parser.add_argument(
            '--model', type=str, metavar='PREFIX',
            help='prefix of model files')
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

    def compute_loss(xs, hs, cs):
        pred_ys, hs, cs = model(xs, hs, cs)
        pred_ys = F.stack(pred_ys, axis=0)
        loss = F.softmax_cross_entropy(
                F.reshape(
                    pred_ys[:,:-1],
                    (pred_ys.shape[0]*(pred_ys.shape[1]-1), pred_ys.shape[2])),
                F.flatten(xs[:,1:]))
        return loss, pred_ys, hs, cs
 
    n_batches = 0
    train_hs, train_cs = None, None
    while True:
        t0 = time.time()
        batch = read_batch()
        model.cleargrads()
        loss, _, train_hs, train_cs = compute_loss(batch, train_hs, train_cs)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        print('TRAIN',
              n_batches,
              cuda.to_cpu(loss.data)/math.log(2),
              time.time()-t0,
              flush=True)

        n_batches += 1
        best_heldout_loss = float('inf')
        if n_batches % 250 == 0:
            heldout_loss = 0.0
            t0 = time.time()
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    hs, cs = None, None
                    for batch in heldout_batches:
                        loss, _, hs, cs = compute_loss(batch, hs, cs)
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

