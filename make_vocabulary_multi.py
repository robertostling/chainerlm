import sys
import pickle
from collections import Counter
import random
import argparse

def main():
    parser = argparse.ArgumentParser(
            description='Vocabulary creation for multilingual model')
    parser.add_argument(
            '--corpus', type=str, metavar='FILE', required=True,
            help='UTF-8 encoded file in language<tab>text format')
    parser.add_argument(
            '--vocabulary', type=str, metavar='FILE', required=True,
            help='pickled character vocabulary file')
    parser.add_argument(
            '--exclude', type=str, metavar='TOKENS',
            help='comma-separated list of tokens to exclude')
    parser.add_argument(
            '--vocabulary-size', type=int, metavar='N',
            help='maximum vocabulary size')
    parser.add_argument(
            '--min-frequency', type=int, metavar='N',
            help='minimum frequency of vocabulary items')
    parser.add_argument(
            '--tokenized', action='store_true',
            help='assume tokenized text (default: character level)')
    args = parser.parse_args()

    corpus_file = args.corpus
    vocab_file = args.vocabulary
    vocab_size = args.vocabulary_size
    min_freq = args.min_frequency
    tokenized = args.tokenized
    exclude = None if args.exclude is None else set(args.exclude.split(','))

    size = 0
    counts = Counter()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.split('\t')
            if len(fields) == 2:
                sentence = fields[1].strip()
                if tokenized:
                    sentence = sentence.split()
                counts.update(sentence)

    print('Creating vocabulary...', flush=True)
    counts = sorted(counts.items(), key=lambda t: t[::-1], reverse=True)
    if min_freq is not None:
        counts = [(c,n) for c,n in counts if n >= min_freq]
    if exclude is not None:
        counts = [(c,n) for c,n in counts if c not in exclude]
    if vocab_size is not None:
        if len(counts) > vocab_size:
            print('Cutoff frequency: %d' % counts[vocab_size][1])
        counts = counts[:vocab_size]
    vocab = {c for c,n in counts}
    vocab = sorted(vocab - {'\t', '\n', '\xa0'})
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f, -1)
    print('Vocabulary written,', len(vocab), 'items')
    for c, n in counts:
        print(c, n)

if __name__ == '__main__': main()

