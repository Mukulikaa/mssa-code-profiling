from sklearn.decomposition import IncrementalPCA

# step 5: Principal component analysis

def pca(a, n_components=2):
    """
    Parameters
    ----------
    a : array_like
        Training data
    n_components : int
        Number of components to keep. Default is 2.
    """
    pca = IncrementalPCA(n_components=n_components)
    pca.fit(a)
    return pca.transform(a)