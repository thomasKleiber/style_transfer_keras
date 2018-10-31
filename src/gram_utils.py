import numpy as np

def vec_to_gramm(vec, IW):
    F = len(vec)
    G = np.zeros((F,F))
    for i in range(F):
        for j in range(F):
            G[i,j] = vec[i]*vec[j]*IW
    return G

def gram_to_vec(gram, IW):
    Gm0 = np.squeeze(gram)
    NF = Gm0.shape[0]
    Vp0 = np.zeros((NF))
    for i in range(NF):
        Vp0[i] = np.sqrt(Gm0[i][i] / IW)
    return Vp0

def dist(g1, g2):
    return np.sum(np.abs(g1-g2)) / np.sum(g1+g2) / 2

def dist_diag(g, IW):
    V = gram_to_vec(g, IW)
    g2 = vec_to_gramm(V, IW)
    return dist(g, g2)
