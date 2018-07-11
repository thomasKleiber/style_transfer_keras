import os
import sys
import glob
from tqdm import *
import time
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



class stylePyr():
    ############################################################################

    ## init image
    radom_input=False
    im_name='toulouse'
    im_path='/home/thomas/Desktop/'
    im_extension='.JPG'

    content_img_init_rescale=1 # 1 = no effect 
    init_noise_amount=0.0

    ## style image selection
    style_images_folder = '/home/thomas/Desktop/chinise/'
    use_full_im_folder=False # overrides im_set below
    im_set = [0]
     
    ## load
    HW=1000

    ## algo setup
    style_layers = ['block2_conv2','block3_conv1']

    pyrdowns = [ 1, 1, 1, 1 ]

    learning_rates = [[5,2], [0.5, 1.1], [0.05, 0]] # mettre le dernier à 0
    
    ITS=2000

    log_granularity=10

    out_folder='./out/'
    out_name = '' # will be built with imname + layers if ''
    run_cnt=0

    
    ################################################################################
    ## internals 
    inpu_im = ''
    pc = 1         # pyramid count
    po = [0,0,0,0] # pyramid offsets

    X = list()
    pimi = list()
    preds = list()
    grams = list()
    
    ################################################################################
    def __init__(self):
        print("initialized!")

    ################################################################################
    def clear(self):
        K.clear_session()
        self.X = list()
        self.pimi = list()
        self.preds = list()
        self.grams = list()
        self.inpu_im = self.im_path + self.im_name + self.im_extension
        self.pc = 1
        self.po = [0, 0, 0, 0]

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
        imgs=glob.glob(self.style_images_folder + '*.jpg')
        imgs.sort()
        if self.use_full_im_folder:
            self.im_set=range(len(imgs))

        for n, id_ in enumerate(imgs):
            img = imageio.imread(id_)
            img = np.float32(img)
            W,H,_=img.shape
            # img = img[W//4:3*W//4, H//4:3*H//4]
            img = resize(img, (self.HW,self.HW), interpolation=INTER_CUBIC)
            self.X.append(img)

        for i in range(len(self.im_set)):
            imi=preprocess_image(self.X[self.im_set[i]])
            self.pimi.append(imi)
        self.pimi=np.array(self.pimi)


    ################################################################################
    def get_gramms_target(self):
        config = tf.ConfigProto()
        session=tf.Session(config=config)
        set_session(session)
        init_op = tf.initialize_all_variables()

        vgg16 = VGG16(weights='imagenet', include_top=False)

        input_img = Input(shape=((None,None,3)))
        x0 = AveragePooling2D((2, 2), padding='same')(input_img)
        x1 = AveragePooling2D((2, 2), padding='same')(x0)
        x2 = AveragePooling2D((2, 2), padding='same')(x1)
        x3 = AveragePooling2D((2, 2), padding='same')(x2)
        x4 = AveragePooling2D((2, 2), padding='same')(x3)
        x5 = AveragePooling2D((2, 2), padding='same')(x4)
        x6 = AveragePooling2D((2, 2), padding='same')(x5)

        if self.pyrdowns[0] > 0:
            self.po[0] = self.pc
            self.pc = self.pc + 1
            vgg16_pd0 = VGG16(input_tensor=x0, weights='imagenet', include_top=False)
        if self.pyrdowns[1] > 0:
            self.po[1] = self.pc
            self.pc = self.pc + 1
            vgg16_pd1 = VGG16(input_tensor=x1, weights='imagenet', include_top=False)
        if self.pyrdowns[2] > 0:
            self.po[2] = self.pc
            self.pc = self.pc + 1
            vgg16_pd2 = VGG16(input_tensor=x2, weights='imagenet', include_top=False)
        if self.pyrdowns[3] > 0:
            self.po[3] = self.pc
            self.pc = self.pc + 1
            vgg16_pd3 = VGG16(input_tensor=x3, weights='imagenet', include_top=False)

        for layer_name in self.style_layers:
            mx = Model(input=vgg16.input, outputs=vgg16.get_layer(layer_name).output)

            pred = mx.predict(self.pimi)
            self.preds.append(pred)

            if s.pyrdowns[0] > 0:
                mx = Model(input=input_img, outputs=vgg16_pd0.get_layer(layer_name).output)
                pred = mx.predict(self.pimi)
                self.preds.append(pred)
            if s.pyrdowns[1] > 0:
                mx = Model(input=input_img, outputs=vgg16_pd1.get_layer(layer_name).output)
                pred = mx.predict(self.pimi)
                self.preds.append(pred)
            if s.pyrdowns[2] > 0:
                mx = Model(input=input_img, outputs=vgg16_pd2.get_layer(layer_name).output)
                pred = mx.predict(self.pimi)
                self.preds.append(pred)
            if s.pyrdowns[3] > 0:
                mx = Model(input=input_img, outputs=vgg16_pd3.get_layer(layer_name).output)
                pred = mx.predict(self.pimi)
                self.preds.append(pred)

        for n, p in enumerate(self.preds):
            GI=list()
            for i in range(p.shape[0]):
                G=gram_matrix(np.squeeze(p[i,:,:,:]))
                GI.append(G.eval(session=session))
            self.grams.append(np.array(GI))
        K.clear_session()

    ################################################################################
    def gram_loss(self, i, j, G, chw):
        """ i index in self.grams, j image index """
        num = K.sum(K.square(np.squeeze(self.grams[i][j])-G)) 
        den = (len(self.im_set) * 4. * ((chw)**2))
        return num / den
        

    ################################################################################
    def iterate(self):
        if self.radom_input:
            np.random.seed(int(time.time()))
            stylized_img_tensor = K.variable(np.random.normal(size=(1, self.HW, self.HW, 3), loc=0., scale=0.1))
        else:
            content_img_raw = imageio.imread(self.inpu_im)
            content_img_raw = resize_image(content_img_raw, target_size=(self.HW,self.HW))
            content_img = preprocess_image_expand(content_img_raw)
            content_img *= self.content_img_init_rescale
            content_img += np.random.normal(loc=0.0, scale=self.init_noise_amount, size=content_img.shape) 
                
            content_img_tensor = K.constant(content_img)
            stylized_img_tensor = K.variable(content_img_tensor)


        ################################################################################
        input_tensor = K.concatenate([stylized_img_tensor], axis=0)
        vgg16 = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

        input_img = Input(tensor=input_tensor)
        x0 = AveragePooling2D((2, 2), padding='same')(input_img)
        x1 = AveragePooling2D((2, 2), padding='same')(x0)
        x2 = AveragePooling2D((2, 2), padding='same')(x1)
        x3 = AveragePooling2D((2, 2), padding='same')(x2)
        x4 = AveragePooling2D((2, 2), padding='same')(x3)
        x5 = AveragePooling2D((2, 2), padding='same')(x4)
        x6 = AveragePooling2D((2, 2), padding='same')(x5)

        #vgg16_ = VGG16(weights='imagenet', include_top=False)
        #mx = Model(input=vgg16_.input, outputs=vgg16.get_layer(self.style_layers[-1]).output)

        vgg16_pd0 = VGG16(input_tensor=x0, weights='imagenet', include_top=False)
        vgg16_pd1 = VGG16(input_tensor=x1, weights='imagenet', include_top=False)
        vgg16_pd2 = VGG16(input_tensor=x2, weights='imagenet', include_top=False)
        vgg16_pd3 = VGG16(input_tensor=x3, weights='imagenet', include_top=False)



        style_loss = K.variable(0.)
        style_loss_pd0 = K.variable(0.)
        style_loss_pd1 = K.variable(0.)
        style_loss_pd2 = K.variable(0.)
        style_loss_pd3 = K.variable(0.)
        for L in range(len(self.style_layers)):
            layer_features = vgg16.get_layer(self.style_layers[L]).output
            generated_features = layer_features[0, :, :, :]
            b, h, w, c = layer_features.shape.as_list()
            G = gram_matrix(generated_features)
            for i in range(len(self.im_set)):
                style_loss = style_loss + self.gram_loss(self.pc*L, i, G, c*h*w)

            if self.pyrdowns[0] > 0:
                layer_features = vgg16_pd0.get_layer(self.style_layers[L]).output
                generated_features = layer_features[0, :, :, :]
                b, h, w, c = layer_features.shape.as_list()
                G = gram_matrix(generated_features)
                for i in range(len(self.im_set)):
                    style_loss_pd0 = style_loss_pd0 + self.gram_loss(self.pc*L + self.po[0], i, G, c*h*w)
            
            if self.pyrdowns[1] > 0:
                layer_features = vgg16_pd1.get_layer(self.style_layers[L]).output
                generated_features = layer_features[0, :, :, :]
                b, h, w, c = layer_features.shape.as_list()
                G = gram_matrix(generated_features)
                for i in range(len(self.im_set)):
                    style_loss_pd1 = style_loss_pd1 + self.gram_loss(self.pc*L + self.po[1], i, G, c*h*w)
            
            if self.pyrdowns[2] > 0:
                layer_features = vgg16_pd2.get_layer(self.style_layers[L]).output
                generated_features = layer_features[0, :, :, :]
                b, h, w, c = layer_features.shape.as_list()
                G = gram_matrix(generated_features)
                for i in range(len(self.im_set)):
                    style_loss_pd2 = style_loss_pd2 + self.gram_loss(self.pc*L + self.po[2], i, G, c*h*w)
                
            if self.pyrdowns[3] > 0:
                layer_features = vgg16_pd3.get_layer(self.style_layers[L]).output
                generated_features = layer_features[0, :, :, :]
                b, h, w, c = layer_features.shape.as_list()
                G = gram_matrix(generated_features)
                for i in range(len(self.im_set)):
                    style_loss_pd3 = style_loss_pd3 + self.gram_loss(self.pc*L + self.po[3], i, G, c*h*w)



        ################################################################################
        alpha=1
        loss = alpha*style_loss + self.pyrdowns[0]*style_loss_pd0 \
            + self.pyrdowns[1]*style_loss_pd1 \
            + self.pyrdowns[2]*style_loss_pd2 \
            + self.pyrdowns[3]*style_loss_pd3
        style_loss_pd = style_loss_pd0+style_loss_pd1+style_loss_pd2+style_loss_pd3

        opt = Adam(lr=(self.learning_rates[0][0]))
        lr_idx=0
        updates = opt.get_updates([stylized_img_tensor], {}, loss)
        to_return = [loss, style_loss, style_loss_pd, stylized_img_tensor]
        train_step = K.function([], to_return, updates)


        ################################################################################
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)

        ################################################################################
        last_loss=np.Inf
        last_loss_pd=np.Inf

        for i in range(self.ITS):
            tic()
            outputs = train_step([])
            if (i % self.log_granularity) == 0:
                lp = last_loss/outputs[1]
                lpd = last_loss_pd/outputs[2]
                print("%-3d : %s (%.3e, %.2f / %.3e, %.2f)" 
                    % (i, toc(), outputs[1], lp, outputs[2], lpd))

                if (lp < 1) and (np.isnan(lpd) or (lpd < 1)):
                    break

                if (lp < self.learning_rates[lr_idx][1]) and (lpd < self.learning_rates[lr_idx][1]) :
                    lr_idx = lr_idx+1
                    print("switching to lr=%.0e" % (self.learning_rates[lr_idx][0]))
                    opt.lr = self.learning_rates[lr_idx][0]

                last_loss = outputs[1] 
                last_loss_pd = outputs[2] 

                stylized_img = K.get_value(stylized_img_tensor)
                stylized_img = deprocess_image(stylized_img)

                imageio.imwrite(self.out_folder + '/' + 'im.jpg', stylized_img)
        
        if self.out_name == '':
            imset_str = str(self.im_set[0])
            for ii in range(len(self.im_set)-1):
                imset_str = imset_str + '_' + str(self.im_set[i+1])
            imname = str(self.run_cnt) + '_' + self.im_name + '_' + self.style_images_folder.split('/')[-2] + '_' + imset_str +  '.jpg'
        else:
            imname = str(self.run_cnt) + '_' + self.out_name + '.jpg'
        imageio.imwrite(self.out_folder + '/' + imname, stylized_img)
        print(imname + " saved")
        self.run_cnt = self.run_cnt+1;



    ################################################################################
    def save_bilan(self):
        fig, ax = plt.subplots( nrows=1, ncols=3, figsize=(150,50) )

        ax[0].imshow(imageio.imread(imgs[self.im_set[0]]))
        ax[1].imshow(stylized_img)
        ax[2].imshow(content_img_raw)
        fig.savefig('bilan.png')
        plt.close(fig)


    def doit(self):
        self.clear()
        self.load_images()
        self.get_gramms_target()
        self.iterate()

    def doitagain(self):
        self.get_gramms_target()
        self.iterate()







################################################################################
################################################################################
################################################################################

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

def gram_matrix(x):
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
