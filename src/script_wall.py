
out_dir = '/home/thomas/work/github/style_transfer_keras/images/'

if True or not 's' in globals():
    exec(open('bryks.py').read())
    exec(open('models_stock.py').read())
    exec(open('style_bryk.py').read())
    exec(open('style_tree.py').read())
    exec(open('style_utils.py').read())
    exec(open('models_stock.py').read())
    s=styleWall(out_dir + '33')

# HW vs OOM:
# 1 pd, 1 layer = max 2550. -50 par layer


s.style_layers = [
        'block2_conv1',
        'block3_conv1',
      #  'block4_conv1',
        ]
s.pyrdowns=[0, 1]
s.HW=1500
s.radom_input=False
s.form_factor=3/2
s.ITS=1000

s.style_images_folder = '/home/thomas/Desktop/allstyl/'
s.base_img_dir='/home/thomas/Desktop/tk/'

for i in range(1000):
    s.create_tmp_folder()
    s.select_random_styl()
    s.select_random_styl(append=True)
    s.select_random_base()
    s.doit(stop_mode='best', overwrite_tmp=False)


