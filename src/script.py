
exec(open("./style_transfer.py").read())
s=stylePyr()

###############################################################################
s.input_img='/home/thomas/Desktop/corse.JPG'
s.content_img_init_rescale=1 
s.init_noise_amount=0.0

s.use_full_im_folder=False
s.im_set = [9]

s.style_layers = ['block2_conv1', 'block3_conv1']
pyrdowns = [ 1, 1, 1, 0 ]

learning_rates = [[5,2], [0.5, 1.1], [0.05, 0]]
s.log_granularity=10

s.out_name = ''
s.run_cnt=54
###############################################################################


s.style_images_folder = '/home/thomas/Desktop/robiac/Mosaic_-_Mosquée_de_Paris.jpg'
s.HW=2000
s.ITS=2000

s.out_folder='/home/thomas/work/jupyter/batch_robiac/'

imgs_styl=glob.glob(s.style_images_folder + '*')
# s.radom_input=True
# for n in range(0,len(imgs_styl)):
#     print("------ %d -------" %(n))
#     s.im_set=[n]
#     s.doit()


s.radom_input=False
imgs_content=glob.glob('/home/thomas/Desktop/content/*')

for i in range(0,1000):
    s.input_img = imgs_content[random.randint(0,len(imgs_content)-1)]
    s.im_set = [random.randint(0, len(imgs_styl)-1)]
    s.doit()

