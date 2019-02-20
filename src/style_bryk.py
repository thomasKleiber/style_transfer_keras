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
    im_set = [2]

    ## load
    HW = 500

    ## algo setup
    style_layers = ['block3_conv2']

    pyrdowns = [ 0 ]

    lrs = [[5,2], [0.5, 1.1], [0.05,0]] # mettre le dernier à 0
    #lrs = [ [0.1, 0]] # mettre le dernier à 0

    ITS = 2000

    log_granularity=20

    out_folder = './out/'
    out_name = ''

    vggs = models_stock()


    ############################################################################
    def __init__(self, out_folder=''):
        if out_folder != '':
            self.out_folder = out_folder
        self._prepare_output()
        self.init_run_cnt()

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

    def select_styl(self, needle, append=False):
        imgs = self._get_style_im_list(filtered=False)
        candidates = [ n for n, i in enumerate(imgs) if needle in basename(i) ]
        if len(candidates) == 1:
            self.printk('selected style : %s%s' % (basename(imgs[candidates[0]]),
                                            " +" if append else ""))
            if not append: self.im_set = candidates
            else: self.im_set.append(candidates[0])
        else:
            self.printk('scanning %s...' % self.style_images_folder)
            if len(candidates) == 0:
                self.printk('no natch found')
            else:
                self.printk('choose :')
                self.list_style_folder(needle=needle)

    def select_random_styl(self, append=False):
        imgs = self._get_style_im_list(filtered=False)
        idx = random.randint(0,len(imgs)-1)
        self.printk('selected rnd style : %s%s' % \
                (imgs[idx], (" (+)" if append else "")))
        if not append:
            self.im_set = [idx]
        else:
            self.im_set.append(idx)

    def select_base(self, needle):
        imgs = self._get_base_im_list()
        candidates = [ basename(i) for i in imgs if needle in basename(i) ]
        if len(candidates) == 1:
            self.printk('selected base : ' + candidates[0])
            self.base_img_name = candidates[0]
        else:
            self.printk('scanning %s...' % self.base_img_dir)
            if len(candidates) == 0:
                self.printk('no natch found')
            else:
                self.printk('choose :')
                self.list_base_folder(needle)

    def select_random_base(self):
        imgs = self._get_base_im_list()
        idx = random.randint(0,len(imgs)-1)
        self.printk('selected rnd base : %s' % (imgs[idx]))
        self.base_img_name = basename(imgs[idx])

    ############################################################################
    def init_run_cnt(self):
        self.run_cnt = 0
        e = os.listdir(self.out_folder)
        while 'tmp%d' % self.run_cnt in e or '%d_im.png' in e:
            self.run_cnt += 1

    def create_tmp_folder(self):
        self.tmp_folder = self.out_folder + '/tmp'
        self.tmp_tmp = self.tmp_folder + '/tmp'
        if 'tmp' in os.listdir(self.out_folder):
            files=[f for f in os.listdir(self.tmp_folder) \
                                                    if not f.startswith('.')]
            if len(files) != 0:
                self.save_tmp_folder()
        os.system('mkdir -p ' + self.tmp_folder)
        os.system('rm -rf ' + self.tmp_folder + '/*')
        os.system('cp ' + self.out_folder + '/.directory ' + self.tmp_folder)

        os.system('mkdir -p ' + self.tmp_tmp)
        os.system('cp ' + self.out_folder + '/.directory ' + self.tmp_tmp)

    def copy_chosen_images(self):
        imgs = self._get_style_im_list(filtered=False)
        for n, i in enumerate(self.im_set):
            os.system('cp "' + imgs[i] + '" "'
                        + self.tmp_folder + '/i%d_%s"' % (n, basename(imgs[i])))
        base = self.base_img_dir + '/' + self.base_img_name
        os.system('cp "' + base + '" "'
                        + self.tmp_folder + '/b_' + self.base_img_name + '"')

    def save_tmp_folder(self):
        os.system('find ' + self.tmp_tmp +
                    ' -regex \'.*[^1]\.png\' -exec rm {} \;')
        cmd = 'mv %s %s%d' % (self.tmp_folder, self.tmp_folder, self.run_cnt)
        os.system(cmd)

    ############################################################################
    def printk(self, buf):
        print(buf)
        os.system('echo "' + buf + '" >> ' + self.tmp_tmp + '/log.txt')

    ############################################################################
    def _get_bryks(self):
        self.bryks = []
        self.brykd = {}
        imgs = self._get_style_im_list(filtered=True)
        for im in imgs:
            for L in self.style_layers:
                for p in self.pyrdowns:
                    b = bryk(im, L, p, self.HW, self.vggs)
                    self.bryks.append(b)
                    k = '%s%d' % (L, p)
                    if not k in self.brykd: self.brykd[k] = [b]
                    else: self.brykd[k].append(b)


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
    def _randomize_styl(self, method='alternate'):
        if not hasattr(self, 'rnd_idx'): self.rnd_idx = 0
        N = len(self.style_layers)*len(self.pyrdowns)
        if method == 'alternate':
            self.rnd_idx += 1
            self.rnd_idx %= N
        if method == 'rnd_each_batch':
            self.rnd_idx = random.randint(0, N-1)
        for k in self.brykd:
            if method == 'rnd_each_bryk':
                self.rnd_idx = random.randint(0, N-1)
            for n, b in enumerate(self.brykd[k]):
                #b.set_loss_factor(1 if n == self.rnd_idx else 0)
                b.set_loss_factor(1 if n == self.rnd_idx else 0)

    ############################################################################
    def get_bryk_initial_losses(self, again=False):
        _, train_step, _ = self._setup_iterations()
        outputs = train_step([])
        return outputs[2:]

    ############################################################################
    def iterate(self, again=False, save=True, stop_mode='dflt',
            overwrite_tmp=True):
        try:
            stylized, step, opt = self._setup_iterations(again)
        except KeyboardInterrupt:
            print('\nKeyboard Interrupt!')
            return
        if not again:
            self.losses = []
            self.last_loss = np.Inf
            self.best_loss = np.Inf
            self.lr_idx=0
            self._increment_final_img_counter()
        self.list_bryks()

        for i in range(self.ITS):
            tic()
            try:
                self._randomize_styl()
                outputs = step([])
            except KeyboardInterrupt:
                self.printk('\nKeyboard Interrupt!')
                save = False
                break

            if (i % self.log_granularity) == 0:
                self._monitor_bryk_losses(outputs)
                li = self.bryks_losses()
                self.losses.append(li)
                curr_loss = np.sum(np.array([u for u in li if u == u]))

                lp = self.last_loss / curr_loss

                if (stop_mode == 'dflt' and
                        lp < 1 and i > 30 and self.lr_idx == (len(self.lrs)-1)):
                    break
                sep = ':'
                self.printk("%-3d %s %s %.3f %.3e || %s"
                    % (i, sep, toc(), lp, curr_loss, self.bryks_losses_log()))
                self.plot_bryks_losses(mode='save')

                if lp < self.lrs[self.lr_idx][1] :
                    self.lr_idx = self.lr_idx+1
                    self.printk("switching to lr=%.0e" % \
                            (self.lrs[self.lr_idx][0]))
                    opt.lr = self.lrs[self.lr_idx][0]

                self.last_loss = curr_loss
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
        self.printk("--------------------------")
        for i, n in enumerate(self.bryks):
            self.printk('<%d> : %s' % (i, n))
        self.printk("--------------------------")

    def bryks_losses(self):
#        return [ b.loss for b in self.bryks if b.active ]
        losses = []
        for b in self.bryks:
            if b.active:
                losses.append(b.loss)
            else:
                losses.append(np.nan)
        return losses

    def bryks_nicknames(self):
        return [ b.nickname() for b in self.bryks if b.active ]

    def plot_bryks_losses(self, mode='show'):
        if not hasattr(self, 'losses'):
            print('no data!'); return
        L = np.array(self.losses)
        # N = self.bryks_nicknames()
        for i in range(L.shape[1]):
            plt.semilogy(L[:,i], label=i)
        plt.grid()
        plt.legend()
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            plt.savefig(self.tmp_folder + '/losses.png', bbox_inches="tight")
            plt.close()


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
        img = self._tensor_to_img(tensor)
        if not overwrite:
            if not hasattr(self, 'tmp_cnt'): self.tmp_cnt = 0
            name = self.tmp_tmp + '/im%03d.png' % self.tmp_cnt
            self.tmp_cnt += 1
            self._save_img(name, img)
        name = self.tmp_folder + '/im.png'
        self._save_img(name, img)

    def _increment_final_img_counter(self):
        self.tmp_cnt = 0
        self.run_cnt = self.run_cnt+1;

    def _save_final_image(self, tensor, verbose=True):
        img = self._tensor_to_img(tensor)
        out_name = self.out_name if self.out_name != '' else 'im'
        imname = str(self.run_cnt) + '_' + out_name + '.png'
        self._save_img(self.out_folder + '/' + imname, img)
        if verbose: self.printk(imname + " saved\n")

    ############################################################################
    def doit(self, save=True, stop_mode='dflt', overwrite_tmp=False):
        self.copy_chosen_images()
        self.vggs.clear()
        self._get_bryks()
        self._load_images()
        self._get_gramms_target()
        self.iterate(save=save, stop_mode=stop_mode,
                        overwrite_tmp=overwrite_tmp)

    def doitagain(self, save=True, stop_mode='dflt', overwrite_tmp=False):
       # self.vggs.clear()
       # self._get_bryks()
       # self._load_images()
       # self._get_gramms_target()
        self.iterate(again=True, save=save, stop_mode=stop_mode,
                        overwrite_tmp=overwrite_tmp)

    def random_images(self, styles=1):
        self.create_tmp_folder()
        self.select_random_styl()
        for i in range(styles-1):
            self.select_random_styl(append=True)
        if not self.radom_input:
            self.select_random_base()
        else:
            self.printk('no base image, random mode set')
        self.copy_chosen_images()

