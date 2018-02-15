import sys
import pickle

import numpy as np
from scipy.cluster.hierarchy import average, dendrogram
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

    if False:
        y = pdist(e, 'cosine')
        z = average(y)
        dendrogram(z, labels=languages)
        plt.show()
    else:
        #m = TSNE().fit_transform(e)
        m = PCA(n_components=2).fit_transform(e)
        plt.scatter(m[:,0], m[:,1])
        for xy, language in zip(m, languages):
            plt.annotate(language, xy)
        plt.show()

if __name__ == '__main__': main()

