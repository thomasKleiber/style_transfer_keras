import os
import gc
import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, AveragePooling2D, Lambda
from keras.models import Model
import tensorflow as tf
#from style_utils import *




def rem_first(dim):
    def func(x):
        if dim == 0:
            return x[:,1:,:,:]
        if dim == 1:
            return x[:,:,1:,:]
        if dim == 2:
            return x[:,1:,1:,:]
    return Lambda(func)

class models_stock:
    def __init__(self):
        self.clear()

    def _model_key(self, in_tensor, pyrdowns, shift):
        return str(in_tensor) + str(pyrdowns) + '_' + str(shift)

    def _check_base_model(self, in_tensor, pyrdowns, shift):
        if pyrdowns > 0:
            k0 = self._model_key(in_tensor, 0, shift)
            if not k0 in self.models:
                self._add_model(in_tensor, 0, shift)

    def _shift_input(self, in_tensor, shift):
        print('shift=%d'%shift)
        if shift == 0:
            return in_tensor
        if shift == 1:
                return rem_first(0)(in_tensor)
        if shift == 2:
            return rem_first(1)(in_tensor)
        return rem_first(2)(in_tensor)

    def _add_model(self, in_tensor, pyrdowns, shift):
        self._check_base_model(in_tensor, pyrdowns, shift)
        key = self._model_key(in_tensor, pyrdowns, shift)
        x = self._shift_input(in_tensor, shift)
        for p in range(pyrdowns):
            x = AveragePooling2D((2,2), padding='same')(x)
        vgg = VGG16(input_tensor=x, weights='imagenet', include_top=False)
        self.models[key] = Model(input=in_tensor, outputs=vgg.layers[-1].output)

    def _get_model(self, in_tensor, key, layer):
        out = self.models[key].get_layer(layer).output
        return Model(input=in_tensor, outputs=out)

    def get(self, in_tensor, pyrdowns, layer, shift):
        key = self._model_key(in_tensor, pyrdowns, shift)
        if key in self.models:
            return self._get_model(in_tensor, key, layer)
        self._add_model(in_tensor, pyrdowns, shift)
        return self.get(in_tensor, pyrdowns, layer, shift)

    def clear(self):
        self.models = {}


