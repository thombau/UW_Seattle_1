# Deep Model Predictive Control
"""
Offline Pre-Learning:
    1. Sparse Autoencoder to train l(b-1)
    2. CRBM to train weights to the hidden layer
    3. One future step prediction (gradients won't be that large)
    4. 10 future steps prediction
    
Online Learning:
    
    
Variables:
    


@author: Thomas Baumeister
"""

import numpy as np
import matplotlib.pyplot as plt
#import time

import tensorflow as tf

from crbm import train_crbm

"""configuration of the Neural Network"""

# Fully connected hidden layer
hidden_layer_size = 20

# Data list for pre-training
data_list = ['motion.mat','speed.mat', 'acceleration.mat', 'angle.mat']

"""
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))
"""

def new_fully_connected_layer(input,            # previous layer
                              num_inputs,       # num. inputs from previous layer
                              num_outputs,      # num outputs
                              weights,          # pre-trained weights
                              #biases,           # pre-trained biases
                              pre_trained=True, # pre-trained weights and biases?
                              use_relu=True):   # use rectified linear units?
    
    # Create weights and biases
    if pre_trained==False:
        weights = tf.get_variable("weights",shape=[num_inputs, num_outputs],
                                  initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases",shape=num_outputs, 
                             initializer=tf.constant_initializer(0.0))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the biases
    layer = tf.matmul(input, weights) + biases

    # activation function
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer    
    

idx = 0
weight_list=[]
for data in data_list:
    crbm, batchdata, W_vgl = train_crbm(dataset=data, training_epochs=5)
    weight_list.append(tf.concat([crbm.B.eval(), crbm.W.eval()],0, name=data))
    
v = batchdata
    
layer_hidden_l_1 = new_fully_connected_layer(input=v,
                                             num_inputs=v.shape[0],
                                             num_outputs=hidden_layer_size,
                                             weights=weight_list[0],
                                             use_relu=True)
    
layer_hidden_l = new_fully_connected_layer(input=v,
                                             num_inputs=v.shape[0],
                                             num_outputs=hidden_layer_size,
                                             weights=weight_list[1],
                                             use_relu=True)