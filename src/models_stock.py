import os
import gc
import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, AveragePooling2D, Lambda
from keras.models import Model
import tensorflow as tf
#from style_utils import *



class models_stock:
    def __init__(self):
        self.clear()

    def _model_key(self, in_tensor, pyrdowns):
        return str(in_tensor) + str(pyrdowns)

    def _check_base_model(self, in_tensor, pyrdowns):
        if pyrdowns > 0:
            k0 = self._model_key(in_tensor, 0)
            if not k0 in self.models:
                self._add_model(in_tensor, 0)

    def _add_model(self, in_tensor, pyrdowns):
        self._check_base_model(in_tensor, pyrdowns)
        key = self._model_key(in_tensor, pyrdowns)
        x = in_tensor
        for p in range(pyrdowns):
            x = AveragePooling2D((2,2), padding='same')(x)
        vgg = VGG16(input_tensor=x, weights='imagenet', include_top=False)
        self.models[key] = Model(input=in_tensor, outputs=vgg.layers[-1].output)

    def _get_model(self, in_tensor, key, layer):
        out = self.models[key].get_layer(layer).output
        return Model(input=in_tensor, outputs=out)

    def get(self, in_tensor, pyrdowns, layer):
        key = self._model_key(in_tensor, pyrdowns)
        if key in self.models:
            return self._get_model(in_tensor, key, layer)
        self._add_model(in_tensor, pyrdowns)
        return self.get(in_tensor, pyrdowns, layer)

    def clear(self):
        self.models = {}


