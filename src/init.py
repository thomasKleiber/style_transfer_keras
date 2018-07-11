exec(open("./style_transfer.py").read())
s=stylePyr()


# Algorithm setup.
# - You *have* to adapt those which description ends with (**)
# - It is fun to play with the ones ending with (*)
# - The other are implementation details (or useless)


## Content Image

# Use random input instead of real image. Bypasses the following 
# "content images" configs. (*)
s.radom_input=True
# Content image (**)
s.input_img='/home/thomas/Desktop/jo.JPG'
# Initial rescale of content image. 1 is "full scale", 0 will give
# similar results as setting random_input to True
s.content_img_init_rescale=1 
# Optionnal additive gaussian noise 
s.init_noise_amount=0.0


## style image(s)

# Style images folder (**)
s.style_images_folder = '/home/thomas/Desktop/misc/'
# use all images in folder (overrides im_set below)
s.use_full_im_folder=False
# list of style images to use (**)
s.im_set = [1,4,5]

## Algorithm setup

# inout image size. This is what you need to tune if you encounter memory
# issues (*)
s.HW=1000
# VGG16 layers to use (*)
s.style_layers = ['block2_conv1', 'block3_conv1']
# Pyrdown factors in loss. Leftmost is 1 pyrdown, then 2, 3, 4. See
# https://en.wikipedia.org/wiki/Pyramid_(image_processing))
pyrdowns = [ 1, 1, 1, 1 ]
# Adaptive learning rate pairs. Each is [learningrate, progress_limit], i.e.
# switch to next lr when progress is less than progress_limit. Progress is 
# progress in loss between two logs. Last value should be 0.
learning_rates = [[5,2], [0.5, 1.1], [0.05, 0]]
# Max number of iterations (the algorithm stops by itself when it doesn't 
# progress any more
s.ITS=4000
# How often a log line will be written, and progress computed (see 
# 'learning_rates descriptions above)
s.log_granularity=10

## Out folder

# where to output images (**)
s.out_folder='/home/thomas/work/jupyter/'
# How to name output image. If '' a name will be built with im_name
s.out_name = ''
# out name will be prefixed with a number, to not to overwrite anything
s.run_cnt=0


