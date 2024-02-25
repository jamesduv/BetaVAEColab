"""
Created on Fri Aug  7 02:11:47 2020

@author: jd
contact: jamesduv@umich.edu
affiliation: University of Michigan, Department of Aerospace Eng., CASLAB
"""
import tensorflow as tf
import numpy as np

class split_output(tf.keras.layers.Layer):
    '''slice a given input tensor in half, return both halves separately'''
    def __init__(self, **kwargs):
        super(split_output, self).__init__()

    def build(self, input_shape):
        super(split_output, self).build(input_shape)

    def call(self, inputs):
        dim = inputs.shape
        target_size = int(dim[1] * 0.5)
        #first slice
        s1 = inputs[:,0:target_size]

        #second slide
        s2 = inputs[:,target_size:]
        output = [s1, s2]
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

class split_tensor(tf.keras.layers.Layer):
    def __init__(self, idx1, idx2, **kwargs):
        super(split_tensor, self).__init__()

    def build(self, input_shape):
        super(split_tensor, self).build(input_shape)

    def call(self, inputs):
        s1 = inputs[:,self.idx1:self.idx2]

        return s1

    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(0.5 * input_shape[1]))
