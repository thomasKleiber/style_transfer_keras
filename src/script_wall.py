

if True or not 's' in globals():
    exec(open('bryks.py').read())
    exec(open('models_stock.py').read())
    exec(open('style_bryk.py').read())
    exec(open('style_tree.py').read())
    exec(open('style_utils.py').read())
    exec(open('models_stock.py').read())
    s=styleWall('32')


s.style_layers = [
        #'block1_conv2',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
       # 'block5_conv1'
        ]

s.style_images_folder = '/home/thomas/Desktop/allstyl/'
s.select_styl('em')
s.radom_input=False
s.base_img_dir='/home/thomas/Desktop/tk/'
s.select_base('da_')

s.HW=2000
s.form_factor=3/2
s.ITS=10000

s.pyrdowns=[0, 1, 2]

def dodo():
    s.pyrdowns=[0, 1, 3]
    s.doit(save=False)
    s.pyrdowns=[0, 1, 2]
    s.doitagain(save=False)
    s.pyrdowns=[0, 1]
    s.doitagain(save=False)
    s.pyrdowns=[0]
    s.doitagain()

#s.paufine(mode=0)

#s.doit(save=False)
#s.paufine(mode=1)

#s.doit(save=False)
#s.paufine(mode=2)

#s.doit(save=False)
#s.paufine(mode=3)
#s.style_images_folder = '/home/thomas/Desktop/paris/'
#for i in range(1,len(s._get_style_im_list(False))):
#    s.im_set=[i]
#    s.doit()
#s.im_set=[24]


#s.pyrdowns=[4]
#s.doit(save=False)
#for p in [3,2,1,0]:
#    s.pyrdowns=[p]
#    s.doitagain(save=False)
#s.style_layers = ['block1_conv1']
#s.doitagain()





#s.style_images_folder = '/home/thomas/Desktop/allimgs/'
#imgs = s._get_style_im_list(filtered=False)

#s.style_images_folder = '/home/thomas/Desktop/allstyl/'
#styls = s._get_style_im_list(filtered=False)
#for i in range(1000):
#    N=random.sample(range(len(imgs)), 1)[0]
#    s.input_img=imgs[N]
#    M=random.sample(range(len(styls)), 1)[0]
#    s.im_set=[M]
#    print('content : %d %s\nstyle : %d %s' % (N,imgs[N], M, styls[M]))
#    s.doit()


#s.HW=2000
#s.doit()

#rslts=[]
#for n, i in enumerate(imgs):
#    print(i)
#    s.vggs.clear()
#    s.im_set=[n]
#    s._get_bryks()
#    s._load_images()
#    s._get_gramms_target()
#    rslts.append(s.get_bryk_initial_losses())

#s.compute_bryk_contribs(ITS=500, HW=256)

#s.get_all_trees(4)
#s.compute_all_bryk_masks()

#s.get_tree(7,4)
#s.compute_bryk_masks(4)
#s.set_bryck_loss_factor(7, 50.)
#s.select_bryk([0,2,4,7])
#for i in range(4):
#    s.select_mask(7, i)
#    s.list_bryks()
#    s.iterate()

#s.iterate()
