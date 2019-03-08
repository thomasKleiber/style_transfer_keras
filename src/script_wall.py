
out_dir = '/home/thomas/work/github/style_transfer_keras/images/'

if True or not 's' in globals():
    exec(open('bryks.py').read())
    exec(open('models_stock.py').read())
    exec(open('style_bryk.py').read())
    exec(open('style_tree.py').read())
    exec(open('style_utils.py').read())
    exec(open('models_stock.py').read())
    s=styleWall(out_dir + '36')

# HW vs OOM:
# 1 pd, 1 layer = max 2550. -50 par layer


s.style_layers = [
        'block2_conv1',
        'block3_conv1',
       # 'block4_conv1',
        ]
s.pyrdowns=[0, 1, 2, 3]
s.HW=1000
s.radom_input=True
s.form_factor=1
s.ITS=2000

s.style_images_folder = '/home/thomas/Desktop/allstyl/'
s.base_img_dir='/home/thomas/Desktop/tk/'


if False:
    for i in range(1000):
        styles=random.randint(1,1)
        Nb=len(s.style_layers)*len(s.pyrdowns)*styles
        s.radom_input = (random.randint(0,9) == 0)
        s.random_images(styles=styles)
        s.doit(stop_mode='toto')
        K.clear_session()




