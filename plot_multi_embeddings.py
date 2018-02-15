import sys
import pickle

import numpy as np
from scipy.cluster.hierarchy import average, ward, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    model = sys.argv[1]
    with open(model + '.metadata', 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        languages = pickle.load(f)

    params = np.load(model + '.npz')
    e = params['language_embeddings/W']
    print(e.shape)

    ignore_languages = set()
    # Uncomment the line below to cherry-pick languages in the same way as
    # Rabinovich et al.
    #ignore_languages = set('HU ET FI EL MT'.split())
    keep = [i for i, language in enumerate(languages)
              if language not in ignore_languages]

    languages = [languages[i] for i in keep]
    e = e[keep,:]

    if True:
        y = pdist(e, 'cosine')
        #z = average(y)
        z = ward(y)
        dendrogram(z, labels=languages)
    else:
        #m = TSNE().fit_transform(e)
        m = PCA(n_components=2).fit_transform(e)
        plt.scatter(m[:,0], m[:,1])
        for xy, language in zip(m, languages):
            plt.annotate(language, xy)

    if len(sys.argv) > 2:
        assert sys.argv[2].endswith('.pdf')
        plt.savefig(sys.argv[2])
    plt.show()

if __name__ == '__main__': main()

