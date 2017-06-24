import sys
import pickle
import zipfile
from collections import Counter
import random

def main():
    corpus_file = sys.argv[1]
    vocab_file = sys.argv[2]
    vocab_size = 255

    size = 0
    counts = Counter()
    with zipfile.ZipFile(corpus_file, 'r') as zipf:
        print('Reading file list...', flush=True)
        files = zipf.infolist()
        random.shuffle(files)
        print('Reading data...', flush=True)
        for info in files:
            if not info.filename.endswith('/'):
                with zipf.open(info.filename) as f:
                    data = str(f.read(), 'utf-8')
                counts.update(data)
                size += len(data)
            if size > 1000000: break

    print('Creating vocabulary...', flush=True)
    counts = sorted(counts.items(), key=lambda t: t[::-1], reverse=True)
    vocab = {c for c,n in counts[:vocab_size]}
    vocab = sorted(vocab - {'\t', '\n', '\xa0'})
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab, f, -1)

if __name__ == '__main__': main()

