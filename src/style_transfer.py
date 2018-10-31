import os
import gc
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
from style_utils import *



class stylePyr():
    ############################################################################

    ## init image
    radom_input=False
    input_img = '/home/thomas/Desktop/toulouse.JPG'

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
    Pooling2D = AveragePooling2D

    learning_rates = [[5,2], [0.5, 1.1], [0.05, 0]] # mettre le dernier à 0
    
    ITS=2000

    log_granularity=10

    out_folder='./out/'
    out_name = '' # will be built with imname + layers if ''
    run_cnt=0

    
    ################################################################################
    ## internals 
    pc = 1            # pyramid count
    po = np.zeros(50) # pyramid offsets

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
        self.pc = 1
        self.po = np.zeros(50)

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
        for i in range(len(s.pyrdowns)):
            if self.pyrdowns[i] > 0:
                self.po[i] = self.pc
                self.pc = self.pc + 1

        for layer_name in self.style_layers:
            mx = Model(input=vgg16.input, outputs=vgg16.get_layer(layer_name).output)
            pred = mx.predict(self.pimi)
            self.append_pred(pred, session)
            x = input_img 
            for i in range(len(s.pyrdowns)):
                x = self.Pooling2D((2,2), padding='same')(x)
                if s.pyrdowns[i] > 0:
                    mx = get_vgg16_extr(input_img, x, layer_name) 
                    pred = mx.predict(self.pimi)
                    self.append_pred(pred, session)

        K.clear_session()

    def append_pred(self, p, session):
        GI=list()
        for i in range(p.shape[0]):
            G = gram_matrix(np.squeeze(p[i,:,:,:]))
            GI.append(G.eval(session=session))
        self.grams.append(np.array(GI))

    ################################################################################
    def gram_loss(self, i, j, G, chw):
        """ i index in self.grams, j image index """
        i=int(i)
        num = K.sum(K.square(np.squeeze(self.grams[i][j])-G)) 
        den = (len(self.im_set) * 4. * ((chw)**2))
        return num / den
        

    ################################################################################
    def iterate(self):

        gc.collect()
        if self.radom_input:
            np.random.seed(int(time.time()))
            stylized_img_tensor = \
                K.variable(np.random.normal(size=(1, self.HW, self.HW, 3),
                                            loc=0., scale=0.1))
        else:
            content_img_raw = imageio.imread(self.input_img)
            content_img_raw = resize_image(content_img_raw, target_size=(self.HW,self.HW))
            content_img = preprocess_image_expand(content_img_raw)
            content_img *= self.content_img_init_rescale
            content_img += np.random.normal(loc=0.0, scale=self.init_noise_amount,
                                            size=content_img.shape) 
                
            content_img_tensor = K.constant(content_img)
            stylized_img_tensor = K.variable(content_img_tensor)


        ################################################################################
        input_tensor = K.concatenate([stylized_img_tensor], axis=0)

        input_img = Input(tensor=input_tensor)
        x=input_img
        vgg16 = get_vgg16_extr(input_img, input_img, self.style_layers[-1])

        vgg16u = []
        style_lossu = []
        for i in range(len(s.pyrdowns)):
            x = self.Pooling2D((2,2), padding='same')(x)
            if s.pyrdowns[i] > 0:
                vgg16u.append(get_vgg16_extr(input_img, x, self.style_layers[-1]))
            else:
                vgg16u.append(None)
            style_lossu.append(K.variable(0.))

        style_loss = K.variable(0.)

        for L in range(len(self.style_layers)):
            layer_features = vgg16.get_layer(self.style_layers[L]).output
            generated_features = layer_features[0, :, :, :]
            b, h, w, c = layer_features.shape.as_list()
            G = gram_matrix(generated_features)
            for i in range(len(self.im_set)):
                style_loss = style_loss \
                    + self.gram_loss(self.pc*L, i, G, c*h*w)

            for i in range(len(s.pyrdowns)): 
                if s.pyrdowns[i] > 0:
                    layer_features = vgg16u[i].get_layer(self.style_layers[L]).output
                    generated_features = layer_features[0, :, :, :]
                    b, h, w, c = layer_features.shape.as_list()
                    G = gram_matrix(generated_features)
                    for j in range(len(self.im_set)):
                        style_lossu[i] = style_lossu[i] \
                            + self.gram_loss(self.pc*L + self.po[i], j, G, c*h*w)

        gc.collect()

        ################################################################################
        loss = style_loss
        style_loss_pd = 0
        for i in range(len(s.pyrdowns)): 
            style_loss_pd = style_loss_pd + self.pyrdowns[i]*style_lossu[i]
        loss = loss + style_loss_pd
        
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

                if (lp < self.learning_rates[lr_idx][1]) \
                        and (lpd < self.learning_rates[lr_idx][1]) :
                    lr_idx = lr_idx+1
                    print("switching to lr=%.0e" % (self.learning_rates[lr_idx][0]))
                    opt.lr = self.learning_rates[lr_idx][0]

                last_loss = outputs[1] 
                last_loss_pd = outputs[2] 

                stylized_img = K.get_value(stylized_img_tensor)
                stylized_img = deprocess_image(stylized_img)

                imageio.imwrite(self.out_folder + '/' + 'im.jpg', stylized_img)
            gc.collect()

        out_name = self.out_name if self.out_name != '' else 'im'
        imname = str(self.run_cnt) + '_' + out_name + '.jpg'
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







