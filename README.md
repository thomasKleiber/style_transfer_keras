# style transfer tool

This toy project is based on a [workshop](https://github.com/JGuillaumin/style-transfer-workshop) by Julien Guillaumin organized by Toulouse Data Science meetup. His page contains some explanations, links and references to papers.

I implemented the following changes:
- Ability to use several style images
- Use multi-scale style analysing
- content image added another way

## dependencies

I use python3, I don't know how python2-friendly it is.

The following packages are used: tensorflow, keras, cv2, scipy, imageio, numpy, matplotlib

## example

This is how I use the code. Not very 'pythonic' but I like it that way

- edit launch.py. Things you need to change (mostly paths) are described

- cd src/

- run your favority python interface. I like [ptpython](https://github.com/jonathanslenders/ptpython)

- run:

```python
# runs the config file. This will create and init a 'style transfer' object called s
>>> exec(open("./launch.py").read())

# you could already run 'as is'
>>> s.doit()
```
- There you are, it should do the job. Then you can tune it on the fly:

```python
# you can change its configs (basically all those of launch.py)
>>> s.im_set = [1,5,8]

# the following command is useful to list style folder content (so that you can
# select what to put in im_set
>>> s.list_im_folder()
--------------------------
 * 0 - 147Lijin.jpg
   1 - 147Lijin_extr.jpg
   2 - 5e687b9e69501f3e719e38bdbde4c9a6.jpg
   3 - huang-junbi.jpg
   4 - maxresdefault.jpg
   5 - rBVaI1lnjJ2AVb5hAAXMrtE-TFk426.jpg
--------------------------
```


