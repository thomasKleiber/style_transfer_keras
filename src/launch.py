
exec(open("./style_transfer.py").read())
s=stylePyr()

s.radom_input=True
# s.im_name='toulouse'
s.im_path='/home/thomas/Desktop/allimgs/'
s.im_extension='.jpg'
# s.content_img_init_rescale=.1 # 1 = no effect 
# s.init_noise_amount=0.0
s.images_folder = '/home/thomas/Desktop/allstyl/'
# s.use_full_im_folder=True # overrides im_set below
s.im_set = [54]
s.HW=1000
# s.style_layers = ['block2_conv1', 'block4_conv1']
# s.do_pyrdown=True # false just forces beta=0
# s.pyrdowns_count=4
# s.learning_rates = [[5,2], [0.5, 1.1], [0.05, 0]] # mettre le dernier à 0
s.ITS=4000
# s.log_granularity=10
s.out_folder='/home/thomas/work/jupyter/'
# s.out_name = '' # will be built with imname + layers if ''
s.run_cnt=0



