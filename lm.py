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

