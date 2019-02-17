import numpy as np
import matplotlib.pyplot as plt


class styleTree():
    ############################################################################
    def __init__(self, FMs):
        self.FMs = FMs
        I, W, H, _ = self.FMs.shape
        self.masks = [[ np.zeros((W, H)) for i in range(I)] for j in range(1)]

    ############################################################################
    def get_km(self, K):
        N = 20000
        sple = []
        I, W, H, _ = self.FMs.shape

        for i in range(N):
            w = np.random.randint(I)
            x = np.random.randint(W)
            y = np.random.randint(H)
            sple.append(np.squeeze(self.FMs[w,x,y,:]))
        self.sple = sple
        self.km, self.groups, self.dists = k_mean(sple, K)

    ############################################################################
    def get_masks(self, verbose=True):
        I, W, H, _ = self.FMs.shape
        K = len(self.km)

        self.masks = [[ np.zeros((W, H)) for i in range(I)] for j in range(K)]
        for h in range(I):
            for i in range(W):
                for j in range(H):
                    g, _ = kmin_distance(self.km, np.squeeze(self.FMs[h,i,j,:]))
                    self.masks[g][h][i, j] = 1

    ############################################################################
    def plot_masks(self):
        _, W, H, _ = self.FMs.shape
        ma = np.zeros((W, H))
        for n, mi in enumerate(self.masks):
            ma = ma + n * mi
        plt.imshow(ma)
        plt.show()

    ############################################################################
    def plot_km_segmentation(self, im):
        W,H = self.masks[0].shape
        im = resize(im, (H,W))
        for i in range(len(self.km)):
            plt.figure()
            plt.imshow(self.masks[i], 'gray', interpolation='none')
        plt.show()



################################################################################
################################################################################
def kmin_distance(km, v):
    mn = np.inf
    imn = -1
    for i, k in enumerate(km):
        dist =  np.sum(np.square(k-v))
        if dist < mn:
            mn = dist
            imn = i
    return imn, mn


################################################################################
def k_mean(rs, k, verbose=True):
    km = []

    for i in range(k):
        km.append(np.random.rand(rs[0].shape[0]))

    groups = np.zeros(len(rs))
    dists = np.zeros(len(rs))

    done = False
    it = 0
    log=''
    while not done:
        log = log + '%3s : ' % str(it)
        it = it + 1
        done = True
        for i in range(len(rs)):
            im, d = kmin_distance(km, rs[i])
            if groups[i] != im: done = False
            groups[i] = im
            dists[i] = d
        for i in range(len(km)):
            grp_idxs = np.equal(groups,i)
            grp_cnt = np.sum(grp_idxs)
            if grp_cnt == 0:
             #   km = km[:i]+km[i+1:]
                pass
            else:
                km[i] = np.zeros(rs[0].shape[0])
                for j, k in enumerate(grp_idxs):
                    if k: km[i] = km[i] + rs[j]
                km[i] = km[i] / grp_cnt
                log = log + "%-5s " % str(grp_cnt)
        if verbose : print(log)
        log=''
    return km, groups, dists

