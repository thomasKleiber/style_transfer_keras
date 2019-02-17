
out_dir = '/home/thomas/work/github/style_transfer_keras/images/'

if True or not 's' in globals():
    exec(open('bryks.py').read())
    exec(open('models_stock.py').read())
    exec(open('style_bryk.py').read())
    exec(open('style_tree.py').read())
    exec(open('style_utils.py').read())
    exec(open('models_stock.py').read())
    s=styleWall(out_dir + '33')

s.style_layers = [
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        ]
s.pyrdowns=[0, 1, 2]
s.HW=2000
s.form_factor=3/2
s.ITS=10000

s.style_images_folder = '/home/thomas/Desktop/allstyl/'
s.select_styl('em')
s.radom_input=False
s.base_img_dir='/home/thomas/Desktop/tk/'
s.select_base('da_')



