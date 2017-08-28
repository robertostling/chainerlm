from chainer import Chain
import chainer.functions as F
import chainer.links as L

class LanguageModel(Chain):
    def __init__(self, alphabet, embedding_size, state_size, n_layers,
                 dropout=0.0):
        self.alphabet = alphabet
        self.alphabet_index = {c:i for i,c in enumerate(alphabet)}
        super().__init__(
                embeddings=L.EmbedID(
                    len(alphabet), embedding_size, ignore_label=-1),
                lstm=L.NStepLSTM(
                    n_layers, embedding_size, state_size, dropout),
                output=L.Linear(
                    state_size, len(alphabet)))

    def __call__(self, xs, hs=None, cs=None):
        hs, cs, ys = self.lstm(hs, cs, list(self.embeddings(xs)))
        return [self.output(y) for y in ys], hs, cs


class MultiLanguageModel(Chain):
    def __init__(self, alphabet, n_languages, language_embedding_size,
                 embedding_size, state_size, n_layers, dropout=0.0):
        self.alphabet = alphabet
        self.alphabet_index = {c:i for i,c in enumerate(alphabet)}
        super().__init__(
                language_embeddings=L.EmbedID(
                    n_languages, language_embedding_size, ignore_label=-1),
                embeddings=L.EmbedID(
                    len(alphabet), embedding_size, ignore_label=-1),
                lstm=L.NStepLSTM(
                    n_layers, language_embedding_size+embedding_size,
                    state_size, dropout),
                output=L.Linear(
                    state_size, len(alphabet)))

    def __call__(self, langs, xs, hs=None, cs=None):
        embedded_xs = list(self.embeddings(xs))
        embedded_langs = self.language_embeddings(langs)
        embedded = [F.concat((x, F.tile(lang, (x.shape[0], 1))))
                    for x, lang in zip(embedded_xs, embedded_langs)]

        hs, cs, ys = self.lstm(hs, cs, embedded)
        return [self.output(y) for y in ys], hs, cs

