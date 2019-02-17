from itertools import *
import random
exec(open("./style_tree.py").read())
exec(open("./style_masked.py").read())
s=styleMasked()


s.style_images_folder = '/home/thomas/Desktop/stylé/'
s.im_set = [9,10,16,19,23]
s.HW = 500
s.ITS = 2000
s.radom_input=False
s.input_img='/home/thomas/Desktop/atlas.JPG'
s.style_layers = ['block3_conv2', 'block4_conv2', 'block5_conv2']

s.clear()
s.load_images()
s.get_style_trees(4)
m = s.ST['block1_conv2'].masks




s.radom_input = False
for i in range(len(m)):
    for j in combinations(range(len(m)),i+1):
        print('---- %s ----' % (str(j)))
        mask = np.zeros(np.array(m[0]).shape)
        for k in j:
            mask = mask + np.array(m[k])
        s.grams=[]
        s.get_gramms_target(mask=mask)
        s.iterate()

for i in range(200):
    j = random.sample(list(range(0,5)),2)
    print('---- %s ----' % (str(j)))
    mask = np.zeros(np.array(m[0]).shape)
    for k in j:
        mask = mask + np.array(m[k])
    s.grams=[]
    s.get_gramms_target(mask=mask)
    s.iterate()


s.HW = 500
s.ITS = 2000
s.run_cnt = 0
for L in [ 'block2_conv2', 'block3_conv2', 'block4_conv2',
        'block5_conv2']:
    s.style_layers = [L]
    s.clear()
    s.load_images()
    s.get_style_trees(4)
    m = s.ST[L].masks
    s.radom_input=True
    for i in range(len(m)):
        s.grams=[]
        s.get_gramms_target(mask=m[i])
        s.out_name = L + '_' + str(i)
        s.iterate()

