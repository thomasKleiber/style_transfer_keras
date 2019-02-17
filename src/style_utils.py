from cv2 import resize
from scipy.misc import imresize
import numpy as np
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.models import Model
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

def resize_keep_scale(img, HW):
    w, h, c = img.shape
    fac = HW / np.sqrt(w*h)
    new_size = (np.int(h*fac), np.int(w*fac))
    # print('%s -> resizing %s to %s' % (HW, str((w,h)) ,str(new_size)))
    return resize(img, new_size)

def size_w_form_factor(HW, ff):
    '''ff : W/H'''
    return (np.int(HW / np.sqrt(ff)), np.int(HW * np.sqrt(ff)))

def deprocess_image(img):

    if len(img.shape)>3:
      img = img[0]
    img += 128

    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gram_matrix(x, mask=None):
    if type(x) == np.ndarray:
        w, h, f = x.shape
    else:
        w, h, f = x.get_shape().as_list()
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

def sci(x, prec=2):
    sign = '-' if x < 0 else ''
    x = np.abs(x)
    exp = np.floor(np.log10(x))
    x = x / 10**exp
    ent = np.floor(x)
    x = x - ent
    dec = np.round(x * 10**prec)

    if prec>0:
        return '%s%d.%de%d' % (sign, ent, dec, exp)
    elif prec==0:
        return '%s%de%d' % (sign, ent, exp)
    else:
        return None
