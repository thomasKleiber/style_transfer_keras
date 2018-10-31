import os
import gc
import sys
import glob
from cv2 import resize, INTER_CUBIC
import random
from scipy import ndimage
from scipy.misc import imsave, imresize, imread
import imageio 
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from tree_helper import *
from style_utils import *


class styleTree():
    ############################################################################

    style_images_folder = '/home/thomas/Desktop/'
    use_full_im_folder=False # overrides im_set below
    im_set = [2]

    FMS = []
    X = []

    def __init__(self, layer):
        self.layer = layer
        print("initialized!")

    ################################################################################
    def list_im_folder(self):
        imgs=glob.glob(self.style_images_folder + '*.jpg')
        imgs.sort()

        print("--------------------------")
        for n, id_ in enumerate(imgs):
            if n in self.im_set:
                tag=' * '
            else:
                tag='   '
            print("%s%d - %s" % (tag, n, os.path.basename(id_)))
        print("--------------------------")
        print("\n")


    ################################################################################
    def load_images(self):
        self.pimi = []
        imgs=glob.glob(self.style_images_folder + '*.jpg')
        imgs.sort()
        if self.use_full_im_folder:
            self.im_set=range(len(imgs))

        for n, id_ in enumerate(imgs):
            if n in self.im_set:
                img = imageio.imread(id_)
                W, H, _ = img.shape
                self.X.append(img)

        for x in self.X:
            imi = preprocess_image(x)
            self.pimi.append(imi)
        self.pimi=np.array(self.pimi)





    ################################################################################
    def get_FMs(self, pim):
        config = tf.ConfigProto()
        session = tf.Session(config=config)
        set_session(session)
        init_op = tf.initialize_all_variables()

        vgg16 = VGG16(weights='imagenet', include_top=False)

        input_img = Input(shape=((None,None,3)))

        mx = Model(input=vgg16.input, outputs=vgg16.get_layer(self.layer).output)
        print(pim.shape)
        self.FMs = mx.predict(pim)
        print('FMs shape : %s' % str(self.FMs.shape))
    

    ################################################################################
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

    ################################################################################
    def get_masks(self, verbose=True):
        I, W, H, _ = self.FMs.shape
        K = len(self.km)

        self.masks = [[ np.zeros((W, H)) for i in range(I)] for j in range(K)]
        for h in range(I):
            for i in range(W):
                for j in range(H):
                    g, _ = kmin_distance(self.km, np.squeeze(self.FMs[h,i,j,:]))
                    self.masks[g][h][i, j] = 1

#        if verbose:
#            print([int(np.sum(self.masks[i])) for i in range(K)])

    ################################################################################
    def plot_masks(self):
        _, W, H, _ = self.FMs.shape
        ma = np.zeros((W, H))
        for n, mi in enumerate(self.masks):
            ma = ma + n * mi
        plt.imshow(ma)
        plt.show()

    ################################################################################
    def plot_km_segmentation(self, im):
        W,H = self.masks[0].shape
        im = resize(im, (H,W))
        for i in range(len(self.km)):
            plt.figure()
           # plt.imshow(im, 'gray', interpolation='none')
            plt.imshow(self.masks[i], 'gray', interpolation='none')
        plt.show()


    ################################################################################
    def go(self):
        self.load_images()
        self.get_FMs(self.pimi)
        #self.mns = []
        #for i in range(2,20):
        self.get_km(10)
        #plt.plot(self.mns)
        plt.show()
        self.get_masks()
        self.plot_masks()
        # self.plot_km_segmentation(self.X[0])




################################################################################
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

