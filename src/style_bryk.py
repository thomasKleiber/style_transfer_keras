import os
import glob
import time
import random
import imageio
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input
import keras.backend as K
from keras.optimizers import Adam, SGD
from os.path import basename
import tensorflow as tf
#from style_utils import *
#from models_stock import *
#from style_tree import *



class styleWall():
    ############################################################################

    ## init image
    radom_input = True
    base_img_dir = '/home/thomas/Desktop/'
    base_img_name = 'toulouse.JPG'
    form_factor = 3/2 # only used if radom_input

    content_img_init_rescale = 1 # 1 = no effect
    init_noise_amount = 0.0

    ## style image selection
    style_images_folder = '/home/thomas/Desktop/'
    use_full_im_folder = False # overrides im_set below
    im_set = [2]

    ## load
    HW = 500

    ## algo setup
    style_layers = ['block3_conv2']

    pyrdowns = [ 0 ]

    lrs = [[5,2], [0.5, 1.1], [0.05, 1.05], [0.05, 0]] # mettre le dernier à 0

    ITS = 2000

    log_granularity=10

    out_folder = './out/'
    out_name = ''
    run_cnt = 0

    vggs = models_stock()


    ############################################################################
    def __init__(self, out_folder=''):
        if out_folder != '':
            self.out_folder = out_folder
        self._prepare_output()

    ############################################################################
    def list_style_folder(self, needle=""):
        imgs = self._get_style_im_list(filtered=False)

        print("--------------------------")
        for n, id_ in enumerate(imgs):
            if n in self.im_set:
                tag=' * '
            else:
                tag='   '
            name = basename(id_)
            if needle in name:  print("%s%d - %s" % (tag, n, name))
        print("--------------------------")
        print("\n")

    def list_base_folder(self, needle=""):
        imgs = self._get_base_im_list()

        print("--------------------------")
        for id_ in imgs:
            name = basename(id_)
            if name == self.base_img_name:
                tag=' * '
            else:
                tag='   '
            if needle in name:  print("%s - %s" % (tag, name))
        print("--------------------------")
        print("\n")

    def _get_style_im_list(self, filtered=True):
        imgs = glob.glob(self.style_images_folder + '*.jpg')
        imgs.sort()
        if not filtered: return imgs
        style_im_list = []
        for n, id_ in enumerate(imgs):
            if n in self.im_set:
               style_im_list.append(id_)
        return style_im_list

    def _get_base_im_list(self):
        imgs = glob.glob(self.base_img_dir + '/*.*')
        imgs.sort()
        return imgs

    def select_styl(self, needle):
        imgs = self._get_style_im_list(filtered=False)
        candidates = [ n for n, i in enumerate(imgs) if needle in basename(i) ]
        if len(candidates) == 1:
            print('selected : ' + basename(imgs[candidates[0]]))
            self.im_set = candidates
        else:
            print('scanning %s...' % s.style_images_folder)
            if len(candidates) == 0:
                print('no natch found')
            else:
                print('choose :')
                self.list_style_folder(needle=needle)

    def select_base(self, needle):
        imgs = self._get_base_im_list()
        candidates = [ basename(i) for i in imgs if needle in basename(i) ]
        if len(candidates) == 1:
            print('selected : ' + candidates[0])
            self.base_img_name = candidates[0]
        else:
            print('scanning %s...' % s.base_img_dir)
            if len(candidates) == 0:
                print('no natch found')
            else:
                print('choose :')
                self.list_base_folder(needle)

    ############################################################################
    def _get_bryks(self):
        self.bryks = []
        imgs = self._get_style_im_list(filtered=True)
        for im in imgs:
            for L in self.style_layers:
                for p in self.pyrdowns:
                    self.bryks.append(bryk(im, L, p, self.HW, self.vggs))

    def _get_shift_bryks(self, mode):
        self.bryks = []
        imgs = self._get_style_im_list(filtered=True)
        for im in imgs:
            self.bryks.append(bryk(im, 'block1_conv1', 0, self.HW, self.vggs,0))
            if mode > 0:
                self.bryks.append(bryk(im, 'block1_conv1', 0, self.HW, self.vggs,1))
                self.bryks.append(bryk(im, 'block1_conv1', 0, self.HW, self.vggs,2))
                self.bryks.append(bryk(im, 'block1_conv1', 0, self.HW, self.vggs,3))
            if mode > 1:
                self.bryks.append(bryk(im, 'block1_conv2', 0, self.HW, self.vggs,1))
                self.bryks.append(bryk(im, 'block1_conv2', 0, self.HW, self.vggs,2))
                self.bryks.append(bryk(im, 'block1_conv2', 0, self.HW, self.vggs,3))
                self.bryks.append(bryk(im, 'block1_conv2', 0, self.HW, self.vggs,0))
            if mode > 2:
                self.bryks.append(bryk(im, 'block2_conv1', 0, self.HW, self.vggs,1))
                self.bryks.append(bryk(im, 'block2_conv1', 0, self.HW, self.vggs,2))
                self.bryks.append(bryk(im, 'block2_conv1', 0, self.HW, self.vggs,3))
                self.bryks.append(bryk(im, 'block2_conv1', 0, self.HW, self.vggs,0))

    def _load_images(self):
        for b in self.bryks : b.open_im()

    def _get_gramms_target(self):
        for b in self.bryks:
            b.get_FMs()
            b.get_gram_tgt()

    ############################################################################
    def _get_input_tensor(self, again):
        if again:
            if not hasattr(self, 'last_image_buffer'):
                print('WARNING : no last buffer...')
            else:
                return K.variable(self.last_image_buffer)
        if self.radom_input:
            return  self._get_random_input_tensor()
        else:
            return self._get_content_input_tensor()

    def _get_random_input_tensor(self):
        size = (1,) + size_w_form_factor(self.HW, self.form_factor) + (3,)
        np.random.seed(int(time.time()))
        return K.variable(np.random.normal(size=size, loc=0., scale=0.1))

    def _get_content_input_tensor(self):
        raw = imageio.imread(self.base_img_dir + self.base_img_name)
        raw = resize_keep_scale(raw, self.HW)
        content_img = preprocess_image_expand(raw)
        content_img *= self.content_img_init_rescale
        content_img += np.random.normal(loc=0.0, scale=self.init_noise_amount,
                                        size=content_img.shape)
        return K.variable(content_img)



    ############################################################################
    def _setup_iterations(self, again=False):
        stylized = self._get_input_tensor(again)
        input_tensor = K.concatenate([stylized], axis=0)
        for b in self.bryks:
            if b.active:
                b.init_iterations(input_tensor)

        loss = K.variable(0.)
        for b in self.bryks:
            if b.active:
                loss += b.get_loss()

        #opt = SGD(lr=(self.lrs[0][0]))
        opt=Adam(lr=(self.lrs[0][0]))
        updates = opt.get_updates([stylized], {}, loss)
        to_return = [loss, stylized]
        for b in self.bryks:
            if b.active:
                to_return += [b.loss_tensor]
        train_step = K.function([], to_return, updates)
        return stylized, train_step, opt

    ############################################################################
    def _monitor_bryk_losses(self, outputs):
        active_idx = 0
        for n, b in enumerate(self.bryks):
            if b.active:
                b.monitor_progress(outputs[active_idx + 2])
                active_idx += 1


    ############################################################################
    def get_bryk_initial_losses(self, again=False):
        _, train_step, _ = self._setup_iterations()
        outputs = train_step([])
        return outputs[2:]

    ############################################################################
    def iterate(self, again=False, save=True, stop_mode='dflt',
            overwrite_tmp=True):
        stylized, train_step, opt = self._setup_iterations(again)
        self.losses = []
        last_loss = np.Inf
        best_loss = np.Inf
        lr_idx=0
        self.list_bryks()
        for i in range(self.ITS):
            tic()
            try:
                outputs = train_step([])
            except KeyboardInterrupt:
                print('\nKeyboard Interrupt!')
                save = False
                break

            if (i % self.log_granularity) == 0:
                self._monitor_bryk_losses(outputs)
                self.losses.append(self.bryks_losses())
                curr_loss = outputs[0]

                lp = last_loss / curr_loss
                print("%-3d : %s %.3f %.3e || %s"
                    % (i, toc(), lp, curr_loss, self.bryks_losses_log()))

                if (stop_mode == 'dflt' and
                        lp < 1 and i > 30 and lr_idx == (len(self.lrs)-1)):
                    break
                if stop_mode == 'best':
                    if curr_loss < best_loss:
                        best_loss = curr_loss
                        self._save_final_image(stylized,
                                verbose=False, increment=False)

                if lp < self.lrs[lr_idx][1] :
                    lr_idx = lr_idx+1
                    print("switching to lr=%.0e" % \
                            (self.lrs[lr_idx][0]))
                    opt.lr = self.lrs[lr_idx][0]

                last_loss = outputs[0]
                self._save_tmp_image(stylized, overwrite=overwrite_tmp)
        if save:
            self._save_final_image(stylized)
        self.last_image_buffer = K.get_value(stylized)


    ############################################################################
    def get_tree(self, b, k):
        self.bryks[b].get_style_tree(k)

    def get_all_trees(self, k):
        for n, b in enumerate(self.bryks):
            print('\n--- %d %s ---' % (n, b))
            b.get_style_tree(k)

    def select_mask(self, bryk_idx, mask_idx, propagate=True):
        self.bryks[bryk_idx].select_mask(mask_idx)
        if propagate:
            for n, b in enumerate(self.bryks):
                if n != bryk_idx:
                    b.set_mask(self.bryks[bryk_idx].get_mask())

    def reset_masks(self):
        for b in self.bryks:
            b.reset_mask()


    ############################################################################
    def list_bryks(self):
        print("--------------------------")
        for i, n in enumerate(self.bryks):
            print('<%d> : %s' % (i, n))
        print("--------------------------")

    def bryks_losses(self):
        return [ b.loss for b in self.bryks if b.active ]

    def bryks_nicknames(self):
        return [ b.nickname() for b in self.bryks if b.active ]

    def plot_bryks_losses(self):
        if not hasattr(self, 'losses'):
            print('no data!'); return
        L = np.array(self.losses)
        N = self.bryks_nicknames()
        for i in range(L.shape[1]):
            plt.semilogy(L[:,i], label=N[i])
        plt.grid()
        plt.legend()
        plt.show()


    def bryks_losses_log(self):
        log = ""
        for n, b in enumerate(self.bryks):
            if b.active:
                log = log + b.progress_log() + '; '
        return log

    def select_all_bryks(self):
        for b in self.bryks:
            b.active = True

    def select_bryk(self, idxs):
        for b in self.bryks:
            b.active = False
        for i in idxs:
            self.bryks[i].active = True

    def set_bryck_loss_factor(self, idx, factor):
        self.bryks[idx].set_loss_factor(factor)




    ############################################################################
    def store_state(self):
        self.stored_state = {
            'ITS' : self.ITS,
            'HW' : self.HW,
            'out_name' : self.out_name,
            'radom_input' : self.radom_input,
            'lrs' : self.lrs,
        }

    def restore_state(self):
        self.ITS = self.stored_state['ITS']
        self.HW = self.stored_state['HW']
        self.out_name = self.stored_state['out_name']
        self.radom_input = self.stored_state['radom_input']
        self.lrs = self.stored_state['lrs']


    ############################################################################
    def bryk_contrib(self, idx, ITS=None, HW=None):
        self.select_bryk([idx])
        self.store_state()
        if ITS is not None: self.ITS = ITS
        if HW is not None: self.HW = HW
        self.out_name = self.bryks[idx].id()
        try:
            self.iterate()
        except KeyboardInterrupt:
            pass
        self.restore_state()
        self.select_all_bryks()

    def compute_bryk_contribs(self, ITS=None, HW=None, idxs=None):
        if idxs is None:
            for n in range(len(self.bryks)):
                self.bryk_contrib(n, ITS, HW)
        else:
            for n in idxs:
                self.bryk_contrib(n, ITS, HW)

    def compute_bryk_masks(self, idx, ITS=None, HW=None):
        self.select_bryk([idx])
        self.store_state()
        if ITS is not None: self.ITS = ITS
        if HW is not None: self.HW = HW
        for m in range(self.bryks[idx].masks_cnt):
            self.select_mask(idx, m, propagate=False)
            self.out_name = self.bryks[idx].id() + '_mask_%d' % m
            try:
                self.iterate()
            except KeyboardInterrupt:
                pass
        self.restore_state()
        self.select_all_bryks()

    def compute_all_bryk_masks(self):
        for n, b in enumerate(self.bryks):
            print('\n--- %d %s ---' % (n, b))
            self.compute_bryk_masks(n)


    ############################################################################
    def _prepare_output(self):
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)

    def _tensor_to_img(self, tensor):
        img = K.get_value(tensor)
        return deprocess_image(img)

    def _save_img(self, path, img):
        imageio.imwrite(path, img, compress_level=0)

    def _save_tmp_image(self, tensor, overwrite=True):
        if not hasattr(self, 'tmp_cnt'):
            self.tmp_cnt = 0
        if overwrite:
            name = self.out_folder + '/im.png'
        else:
            name = self.out_folder + '/im%d.png' % self.tmp_cnt

        self.tmp_cnt += 1
        img = self._tensor_to_img(tensor)
        self._save_img(name, img)

    def _save_final_image(self, tensor, verbose=True, increment=True):
        img = self._tensor_to_img(tensor)
        out_name = self.out_name if self.out_name != '' else 'im'
        imname = str(self.run_cnt) + '_' + out_name + '.png'
        self._save_img(self.out_folder + '/' + imname, img)
      #  imf=cv2.bilateralFilter(img,4,75,75)
      #  imnamef = str(self.run_cnt) + '_' + out_name + 'f.png'
      #  self._save_img(self.out_folder + '/' + imnamef, imf)
        if verbose:
            print(imname + " saved\n")
     #       print(imnamef + " saved\n")
        if increment:
            self.run_cnt = self.run_cnt+1;


    ############################################################################
    def doit(self, save=True):
        self.vggs.clear()
        self._get_bryks()
        self._load_images()
        self._get_gramms_target()
        #self.iterate(save=save)
        self.iterate(save=save)

    def doitagain(self, save=True):
        self.vggs.clear()
        self._get_bryks()
        self._load_images()
        self._get_gramms_target()
        self.iterate(again=True, save=save)

    def paufine(self, mode=1):
        self.vggs.clear()
        self._get_shift_bryks(mode=mode)
        self._load_images()
        self._get_gramms_target()
        self.iterate(again=True, save=True, stop_mode='best', overwrite_tmp=False)


