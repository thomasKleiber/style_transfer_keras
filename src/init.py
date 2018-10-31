# Algorithm setup.
# see doc/Demo.jpynb for a description of these configs
###############################################################################

exec(open("./style_transfer.py").read())
s=stylePyr()

###############################################################################
s.radom_input=True
s.input_img='/home/thomas/Desktop/corse.JPG'
s.content_img_init_rescale=1 
s.init_noise_amount=0.0

s.style_images_folder = '/home/thomas/Desktop/robiac/'
s.use_full_im_folder=False
s.im_set = [1]

s.HW=1000
s.style_layers = ['block2_conv1', 'block3_conv1']
pyrdowns = [ 1, 1, 0, 0 ]

learning_rates = [[5,2], [0.5, 1.1], [0.05, 0]]
s.ITS=4000
s.log_granularity=10

s.out_folder='/home/thomas/work/jupyter/'
s.out_name = ''
s.run_cnt=0
###############################################################################

#s.doit()

