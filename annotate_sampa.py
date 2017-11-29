#!/usr/bin/env python3

import sys
import pickle
import argparse
from collections import defaultdict
import os.path

import numpy as np
from scipy.spatial.distance import cosine

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F

from lm import LanguageModel
from sampacorpus import SAMPACorpus

def main():
    parser = argparse.ArgumentParser(
            description='LSTM language model inferences')
    parser.add_argument(
            '--drop-unknown', action='store_true',
            help='silently drop unknown characters (default: crash)')
    parser.add_argument(
            '--gpu', type=int, metavar='N', default=-1,
            help='gpu to use (default: use CPU)')
    parser.add_argument(
            '--chunk-size', type=int, metavar='N', default=1023,
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

    print(train_args)
    print('Vocabulary:', ' '.join(vocab))

    model = LanguageModel(
            vocab, train_args.embedding_size, train_args.lstm_size,
            train_args.layers)

    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    xp = model.xp

    serializers.load_npz(args.model + '.npz', model)

    # Yield a sequence of overlapping (by one symbol) sequences
    def get_batches(filename):
        sc = SAMPACorpus(filename)
        text = ['.']
        while True:
            try:
                sentence = sc.read_sentence(restart=False)
                for word in sentence[:-1]:
                    text.extend(word)
                    text.append('_')
                text.extend(sentence[-1])
                text.append('.')
            except EOFError:
                break

        if args.drop_unknown:
            symbols = [model.alphabet_index[c] for c in text
                       if c in model.alphabet_index]
        else:
            symbols = [model.alphabet_index[c] for c in text]

        for i in range(0, len(symbols)-1, args.chunk_size):
            chunk = symbols[i:i+args.chunk_size+1]
            yield xp.array([chunk], dtype=xp.int32)

    chainer.config.train = False
    chainer.config.enable_backprop = False

    for filename in args.filenames:
        if not os.path.exists(filename):
            print('Skipping', filename, 'since it does not exist')
            continue
        if os.path.exists(filename+'.entropy'):
            print('Will not overwrite', filename+'.entropy')
            continue
        print('Reading', filename)

        with open(filename+'.entropy', 'w', encoding='utf-8') as f:
            hs, cs = None, None
            for batch in get_batches(filename):
                pred, hs, cs = model(batch[:,:-1], hs, cs)
                pred = F.stack(pred, axis=0)
                xent = F.softmax_cross_entropy(
                        pred[0, :, :], batch[0, 1:],
                        reduce='no')
                pred = cuda.to_cpu(F.softmax(pred[0]).data)[:, :]
                xent = cuda.to_cpu(xent.data) / np.log(2.0)
                ent = -(pred*np.log2(pred)).sum(axis=-1)
                for c, c_xent, c_ent in zip(batch[0, 1:], xent, ent):
                    print('%s %.3f %.3f' % (vocab[c], c_xent, c_ent), file=f)


if __name__ == '__main__': main()

