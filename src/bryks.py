from keras.layers import Input, AveragePooling2D
from scipy.misc import imsave, imresize, imread
from cv2 import resize, INTER_CUBIC
import tensorflow as tf
import imageio
import keras.backend as K
#from style_utils import *
#from style_tree import *
import re


class bryk:
    def __init__(self, im_path, layer, pyrdowns, HW, vggs):
        self.im_path = im_path
        self.layer = layer
        self.pyrdowns = pyrdowns
        self.HW = HW
        self.active = True
        self.mask = None
        self.vggs = vggs

    def activate(self):
        self.active=True
        try:
            self.set_loss_factor(1)
        except Exception as e:
            print('got exception ' + str(e))

    def desactivate(self):
        self.active=False
        try:
            self.set_loss_factor(0)
        except Exception as e:
            print('got exception ' + str(e))

    def __str__(self):
        nm = self.im_path
        m = re.match('.*/(.*)\.[jJ][pP][eE]?[gG]', nm)
        if m: nm = m.group(1)
        return '%s, %s, %d pyrd., %s%s' % \
            (nm, self.layer, self.pyrdowns,
            'ON' if self.active else 'off',
            ', mask %s' % self.mask_str if self.mask is not None else '')

    def id(self):
        nm = self.im_path
        m = re.match('.*/(.*)\.[jJ][pP][eE]?[gG]', nm)
        if m: nm = m.group(1)
        return '%s__%s__%d_pyrd_%d_sh' \
                % (nm, self.layer, self.pyrdowns)

    def nickname(self):
        m = re.match('block([0-9])_conv([0-9])', self.layer)
        if m:
            # return chr(ord('A') + int(m.group(1)) - 1) + \
            #       chr(ord('A') + int(m.group(2)) - 1) + '%d' % self.pyrdowns
            return 'B' + m.group(1) + 'C' + m.group(2) + '_%d' % self.pyrdowns
        return None

    def open_im(self):
        img = imageio.imread(self.im_path)
        img = resize_keep_scale(img, self.HW)
        self.img = preprocess_image_expand(img)

    def get_FMs(self):
        input_img = Input(shape=((None,None,3)))
        x = input_img
        for p in range(self.pyrdowns):
            x = AveragePooling2D((2,2), padding='same')(x)
        self.mx = get_vgg16_extr(input_img, x, self.layer)
        self.FMs = self.mx.predict([self.img])
        self.st = styleTree(self.FMs)


    def get_style_tree(self, k):
        self.masks_cnt = k
        self.st.get_km(k)
        self.st.get_masks()

    def _get_iterable_model(self, input_tensor):
        self.mx = self.vggs.get(input_tensor, self.pyrdowns, self.layer)

    def _init_progress_monitor(self):
        self.last_loss = np.Inf
        self.progress = np.Inf

    def monitor_progress(self, loss):
        self.loss = loss
        self.progress = self.last_loss / loss
        self.last_loss = loss

    def progress_log(self):
        return sci(np.float(self.loss))
        #return "%.0e/%.2f" % (np.float(self.loss), np.float(self.progress))

    def init_iterations(self, input_tensor):
        self._get_iterable_model(input_tensor)
        self._init_progress_monitor()

    def reset_mask(self):
        self.mask = None

    def resize_mask_for_me(self, mask):
        mask = resize(mask, self.st.masks[0][0].shape)
        return np.double(mask != np.zeros(mask.shape))

    def set_mask(self, mask):
        self.mask_str = 'ext.'
        self.mask = self.resize_mask_for_me(mask)
        self.get_gram_tgt()

    def get_mask(self):
        return self.mask

    def select_mask(self, idxs):
        if type(idxs) == int: idxs = [idxs]
        self.mask_str = str(idxs)
        self.mask = np.zeros(self.st.masks[0][0].shape)
        for i in idxs:
            self.mask = self.mask + self.st.masks[i][0]
        self.get_gram_tgt()

    def set_loss_factor(self, factor):
        K.set_value(self.loss_factor, factor*self.active)
#        self.get_gram_tgt()

    def gram_matrix(self, x, mask=None):
        if type(x) == np.ndarray:
            w, h, f = x.shape
        else:
            w, h, f = x.get_shape().as_list()
        if mask is not None:
            W, H = mask.shape
            x = x * mask[:, :, None] * W * H / np.sum(mask)
        features = K.batch_flatten(K.permute_dimensions(x, (2,0,1)))
        gram_matrix = K.dot(features, K.transpose(features))
        return  gram_matrix / (w*h*f)

    def get_gram_tgt(self):
        config = tf.ConfigProto()
        session = tf.Session(config=config)
        set_session(session)
        session.run(tf.global_variables_initializer())
        fms = np.squeeze(self.FMs[0,:,:,:])
        G = self.gram_matrix(fms, self.mask)
        self.gram = G.eval(session=session)
        K.clear_session()

    def get_loss(self):
        self.loss_factor = K.variable(1.)
        fms = self.mx.get_layer(self.layer).output
        G = self.gram_matrix(fms[0,:,:,:])
        self.loss_tensor = K.sum(K.square(self.gram - G))
        return self.loss_tensor * self.loss_factor

