import sys
import pickle

import numpy as np
from scipy.cluster.hierarchy import average, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def main():
    model = sys.argv[1]
    with open(model + '.metadata', 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        languages = pickle.load(f)

    params = np.load(model + '.npz')
    e = params['language_embeddings/W']
    print(e.shape)

    y = pdist(e, 'cosine')
    z = average(y)
    dendrogram(z, labels=languages)
    plt.show()

if __name__ == '__main__': main()

