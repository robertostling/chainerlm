import gzip

SAMPAVocabulary = r'''
i: e: E: y: }: 2: u: o: A:
{: 9: { 9 @ t` d` n` s` l`
x\
p b t d k g
f v s S h C
m n N r l j
I e E Y 2 U O a u0
_ .
'''.split()


class SAMPACorpus:
    def __init__(self, path):
        self.path = path
        self.open_file()

    def open_file(self):
        if self.path.endswith('.gz'):
            self.f = gzip.open(self.path, 'rt', encoding='utf-8')
        else:
            self.f = open(self.path, 'r', encoding='utf-8')

    def read_sentence(self, restart=True):
        sentence = []
        while True:
            for line in self.f:
                morphemes = line.rstrip('\n').split()
                if not morphemes:
                    assert sentence
                    return sentence
                sentence.append(morphemes)
            self.f.close()
            if not restart:
                raise EOFError()
            self.open_file()
            if sentence: return sentence

    def morphemes(self, buf_size, word_separator='_', sentence_separator='.'):
        while True:
            sentences = []
            length = 0
            while length < buf_size:
                sentences.append(self.read_sentence())
                length += len(sentences[-1])
            for sentence in sentences:
                for word in sentence:
                    for phoneme in word:
                        yield phoneme
                    if word_separator: yield word_separator
                if sentence_separator: yield sentence_separator

    def stream(self, size, **kwargs):
        buf = []
        ms = self.morphemes(size*16, **kwargs)
        while True:
            buf = [next(ms) for _ in range(size)]
            yield buf


if __name__ == '__main__':
    import sys
    from pprint import pprint
    sc = SAMPACorpus(sys.argv[1])
    streams = [sc.stream(16) for _ in range(4)]
    for _ in range(2):
        batch = [''.join(next(s)) for s in streams]
        pprint(batch)

