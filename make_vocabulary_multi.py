import sys
import pickle
from collections import Counter
import random

def main():
    corpus_file = sys.argv[1]
    vocab_file = sys.argv[2]
    vocab_size = 255
    if len(sys.argv) > 3:
        vocab_size = int(sys.argv[3])

    size = 0
    counts = Counter()
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            fields = line.split('\t')
            if len(fields) == 2:
                sentence = fields[1].strip()
                counts.update(sentence)

    print('Creating vocabulary...', flush=True)
    counts = sorted(counts.items(), key=lambda t: t[::-1], reverse=True)
    vocab = {c for c,n in counts[:vocab_size]}
    vocab = sorted(vocab - {'\t', '\n', '\xa0'})
    if len(counts) > vocab_size:
        print('Cutoff frequency: %d' % counts[vocab_size][1])
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f, -1)

if __name__ == '__main__': main()

