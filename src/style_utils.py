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
import time


################################################################################
################################################################################
################################################################################

def get_vgg16_extr(input_image, vgg16_input_tensor, layer):
    vgg = VGG16(input_tensor=vgg16_input_tensor, weights='imagenet', include_top=False)
    mx = Model(input=input_image, outputs=vgg.get_layer(layer).output)
    for layer in mx.layers:
        layer.trainable = False
    return mx


def preprocess_image(img):
  img = img.astype(np.float32)
  #img = img[..., ::-1]
  img = img - 128
  return img

def preprocess_image_expand(img):
  img = img.astype(np.float32)
  #img = img[..., ::-1]
  img = img - 128
  img = np.expand_dims(img, axis=0)
  return img

def scale_for_display(img):
    scaled=(img-img.min())/255
    scaled=scaled/scaled.max()
    return scaled

def deprocess_image(img):

    if len(img.shape)>3:
      img = img[0]
    # add the mean (BGR format)
    img += 128

    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gram_matrix(x, mask=None):
    if mask is not None:
        W, H = mask.shape
        x = x * mask[:, :, None] * W * H / (np.sum(mask))
    features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
    gram_matrix = K.dot(features, K.transpose(features))
    return gram_matrix


def resize_image(img, target_size=(256, 256)):
  h, w, _ = img.shape
  short_edge = min([h,w])
  yy = int((h - short_edge) / 2.)
  xx = int((w - short_edge) / 2.)
  img = img[yy: yy + short_edge, xx: xx + short_edge]
  img = imresize(img, size=target_size, interp='bicubic')
  return img

def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    if 'startTime_for_tictoc' in globals():
        return "%.3f s" % (time.time() - startTime_for_tictoc)
    else:
        return "Toc: start time not set"
