
out_dir = '/home/thomas/work/github/style_transfer_keras/images/'

if True or not 's' in globals():
    exec(open('bryks.py').read())
    exec(open('models_stock.py').read())
    exec(open('style_bryk.py').read())
    exec(open('style_tree.py').read())
    exec(open('style_utils.py').read())
    exec(open('models_stock.py').read())
    s=styleWall(out_dir + '34')

# HW vs OOM:
# 1 pd, 1 layer = max 2550. -50 par layer


s.style_layers = [
        'block2_conv1',
        'block3_conv1',
        ]
s.pyrdowns=[0, 1, 2, 3]
s.HW=1500
s.radom_input=False
s.form_factor=3/2
s.ITS=300

s.style_images_folder = '/home/thomas/Desktop/allstyl/'
s.base_img_dir='/home/thomas/Desktop/tk/'


if True:
    for i in range(1000):
        styles=random.randint(1,2)
        Nb=len(s.style_layers)*len(s.pyrdowns)*styles
        s.random_images(styles=styles)
        s.doit(stop_mode='toto')
        s.bryks[random.randint(0, Nb-1)].desactivate()
        s.doitagain(stop_mode='toto')
        s.bryks[random.randint(0, Nb-1)].desactivate()
        s.doitagain(stop_mode='toto')
else:
    s.random_images(styles=2)

