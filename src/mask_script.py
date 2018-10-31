from itertools import *
import random
exec(open("./style_tree.py").read())
exec(open("./style_masked.py").read())
s=styleMasked()

TASK = 'list_styles'
TASK = 'random_sple'
TASK = None

s.style_images_folder = '/home/thomas/Desktop/stylé/'
s.im_set = [9,10,16,19,23]
s.HW = 1000
s.ITS = 500
s.radom_input=False
s.input_img='/home/thomas/Desktop/ponton.JPG'

s.clear()
s.load_images()
s.get_style_trees(8)
m = s.ST['block3_conv2'].masks


if TASK == 'list_styles':
    s.radom_input=True
    for i in range(len(m)):
        s.grams=[]
        s.get_gramms_target(mask=m[i])
        s.iterate()


if TASK == 'random_sple':
    for i in range(200):
        j = random.sample(list(range(0,5)),2)
        print('---- %s ----' % (str(j)))
        mask = np.zeros(np.array(m[0]).shape)
        for k in j:
            mask = mask + np.array(m[k])
        s.grams=[]
        s.get_gramms_target(mask=mask)
        s.iterate()
