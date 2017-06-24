import zipfile
import random

class ZipCorpus:
    def __init__(self, path):
        self.zipfile = zipfile.ZipFile(path)
        self.files = sorted(
                [info.filename for info in self.zipfile.infolist()
                 if not info.filename.endswith('/')])

    def character_stream(self, size, normalize_blank=True, separator=' '):
        data = ''
        buf = ''
        while True:
            while not buf:
                filename = random.choice(self.files)
                with self.zipfile.open(filename) as f:
                    buf = str(f.read(), 'utf-8')
                    if normalize_blank:
                        buf = ' '.join(buf.split())
                    if separator: buf = buf + separator

            while buf and len(data) < size:
                need = size - len(data)
                data = data + buf[:need]
                buf = buf[need:]

            if len(data) == size:
                yield data
                data = ''


def test():
    import sys
    zipcorpus = ZipCorpus(sys.argv[1])
    streams = [zipcorpus.character_stream(72) for _ in range(4)]
    for _ in range(10):
        batch = [next(stream) for stream in streams]
        for chunk in batch:
            print(chunk)
        print('-'*72)

if __name__ == '__main__': test()
