# No more need to know the birefringence K as we use a Variational to 
# inference that value
"""
###############################################################################
##                       Deep Model Predictive Control                       ##
###############################################################################

Task:
    The task is to control an arbitrary dynamical system using this 
    Deep Model Predictive Control architecture:
                                       _    _        _
                                      | |  |l|      | |
                          l_{t-1}--O->|l|->|o|->O-->|o|-->x_{t+1}
                                   ^  |_|  |_|  ^   |_|
                                   |           /|
                                   |       _  / |
                                   |      | |/  |
                               ____|___  /|c|   |
                              |        |/ |_|   |
                             _|__     _|__     _|__
                            |_lp_|   |_lc_|   |_f__|
                             _|__     _|__     _|__
                            /____/   /____/   /____/
                            v_{t-1}  v_{t}    u_{t+1}
        
        Where:
            v:  are the inputs including the states as well as the control 
                inputs of past ({t-1}) and current ({t}) values
            u:  are the control inputs - here these are the predicted future 
                inputs ({t+1})
            lp: hidden layer of the past input v    -> for long term dynamics
            lc: hidden layer of the current input v -> for long term dynamics
            c:  hidden layer of the current input v  -> for current dynamics
            f:  hidden layer of the future input u   -> for future predictions
            l:  latent layer (LSTM cell, capturing long term dynamics)
            lo: hidden layer (input is output of the latent layer)
            o:  output layer (prediction of future states of the system)

Offline Pre-Learning (Model Prediction):
    1. CRBM to train weights to the hidden layer
    2. Five future step prediction (gradients won't be that large)
    3. 10 future steps prediction
    
Online Learning:
    1. Predict the 10 future orientations (angles) of the wave plates and the  
       nine future states of the laser in order to minimize the cost function
    2. Update the weights and biases of the model - continuously improvment of
       the model
    
    
Variables:
    During the Model Prediction step:
        W_lc_past:      past weights of hidden_layer_lc,
                            shape=[delay*num_parameters,hidden_layer_size]
        W_lc_current:   current weights of hidden_layer_lc,
                            shape=[num_parameters,hidden_layer_size]
        B_lc:           biases of hidden_layer_lc,
                            shape=[hidden_layer_size,]
        W_lp_past:      past weights of hidden_layer_lp,
                            shape=[delay*num_parameters,hidden_layer_size]
        W_lp_current:   current weights of hidden_layer_lp,
                            shape=[num_parameters,hidden_layer_size]
        B_lp:           biases of hidden_layer_lp,
                            shape=[hidden_layer_size,]
        W_c_past:       past weights of hidden_layer_c,
                            shape=[delay*num_parameters,hidden_layer_size]
        W_c_current:    current weights of hidden_layer_c,
                            shape=[num_parameters,hidden_layer_size]
        B_c:            biases of hidden_layer_c,
                            shape=[hidden_layer_size,]
        W_f_past:       past weights of hidden_layer_f,
                            shape=[delay*num_wave_plates,hidden_layer_size] 
        W_f_current:    current weights of hidden_layer_f,
                            shape=[num_wave_plates,hidden_layer_size]
        B_f:            biases of hidden_layer_f,
                            shape=[hidden_layer_size,]
        W_l:            weights of latent_layer,
                            shape=[hidden_layer_size,latent_layer_size]
        B_l:            biases of latent_layer,
                            shape=[latent_layer_size,]
        W_lo:           weights of hidden_layer_lo,
                            shape=[latent_layer_size,hidden_layer_size]
        B_lo:           biases of hidden_layer_lo,
                            shape=[hidden_layer_size,]
        W_o:            weights of hidden_layer_o,
                            shape=[hidden_layer_size,num_states]
        B_o:            biases of hidden_layer_o,
                            shape=[num_states,]
        W_ll:           weights between previous latent_layer and actual one,
                            shape=[latent_layer_size,latent_layer_size]
        W_h:            weights if additional hidden layer is implemented,
                            shape=[hidden_layer_size,hidden_layer_size]
        B_h:            biases if additional hidden layer is implemented,
                            shape=[hidden_layer_size,]
    
    Determining the best control inputs:
        control_input:  angles which can be updated during learning,
                            shape=[num_wave_plates, time_steps=10]


@author: Thomas Baumeister
"""
#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow libraries
# The whole graph is build using tensorflow tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.platform import tf_logging as logging
import tensorflow as tf

import os
import sys

# standard mathamatical operations
import numpy as np
#import matplotlib.pyplot as plt
#import time

# import Conditional Restricted Reynold Boltzman function for the first pre
# training step
from crbm import train_crbm

# import the laser simulation - this will be used while determining the control
# inputs in order to compare the predicted and the true states/costfunction
from mlock_CNLS import laser_simulation

# function to import the data to train the model
#from load_preprocess import load_data

# csv package to write results into a CSV file
import csv
import xlrd

import argparse
import operator

FLAGS = None

#%%
"""
###############################################################################
##                               NN Settings                                 ##
###############################################################################
"""
# in order to use different layer sizes, we have to include one further weight
# between latent and output layer
hidden_layer_size = 200
latent_layer_size = 150

# delay for the past inputs
delay = 5
# number of inputs/input types
num_states = 3
num_latent_var = 1
num_wave_plates = 4
# 3 states + 4 wave plates - num_latent_var
num_parameters = num_states + num_wave_plates - num_latent_var
# total number of inputs
num_inputs_RNN = (delay + 1)*(2*num_parameters + num_wave_plates)
# variable for the creation of the RNN Cell to distiguish between the pre 
# training and the control task
threshold = 0


# time steps of different phases of training
steps_phase_1 = 1
steps_phase_2 = 10
steps_phase_crtl = 10

train_batch_size = 100

# network architecture of the Variational Autoencoder (VAE)
vae_neurons = 200
network_architecture = \
    dict(n_hidden_recog_0 = vae_neurons, # 0st layer encoder neurons
         n_hidden_recog_1 = vae_neurons, # 1st layer encoder neurons
         n_hidden_recog_2 = vae_neurons, # 2nd layer encoder neurons
         n_hidden_gener_0 = vae_neurons, # 0st layer decoder neurons
         n_hidden_gener_1 = vae_neurons, # 1st layer decoder neurons
         n_hidden_gener_2 = vae_neurons, # 2nd layer decoder neurons
         n_input=6,                    # states of the system + 4 wave plates
         n_z=2)                        # dimensionality of latent space

# network architecture for mapping K to good control inputs
layer_struct = [1, 5, 7, 9, 6, num_wave_plates]

# number of hidden layers  
layers = 2
    
reuse_model = True
new_dataset = False


""" Pre-set some FLAGS to easily change parameters """

parser = argparse.ArgumentParser()
#parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
#                      default=False,
#                      help='If true, uses fake data for unit testing.')
parser.add_argument('--max_pre_steps', type=int, default=250,
                      help='Number of steps to pre-train using crbm.')
parser.add_argument('--max_steps_vae', type=int, default=51,
                      help='Number of steps to run VAE.')
parser.add_argument('--max_steps_K_u', type=int, default=50000,
                      help='Number of steps to run trainer K_u_mapping.')
parser.add_argument('--max_steps_1', type=int, default=450000,
                      help='Number of steps to run trainer 1.')
parser.add_argument('--max_steps_10', type=int, default=250000,
                      help='Number of steps to run trainer 2.')
parser.add_argument('--learning_rate', type=float, default=0.00001,
                      help='Initial learning rate')
parser.add_argument('--learning_rate_ctrl', type=float, default=0.01,
                      help='Initial learning rate_control')
parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum of the optimizer.')
parser.add_argument('--log_dir', type=str,
                      default='deepMPC/logs/deepMPC_with_summaries',
                      help='Summaries log directory')
parser.add_argument('--save_dir', type=str,
                      default='Save',
                      help='Save trainables directory')
parser.add_argument('--save_vae_dir', type=str,
                      default='Save_vae',
                      help='Save VAE trainables directory')
parser.add_argument('--save_map_dir', type=str,
                      default='Save_map',
                      help='Save map trainables directory')
FLAGS, unparsed = parser.parse_known_args()

#%%

"""
###############################################################################
##                            Create Directories                             ##
###############################################################################
"""
if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
tf.gfile.MakeDirs(FLAGS.log_dir)

if not reuse_model:
    if tf.gfile.Exists(FLAGS.save_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_dir)
    tf.gfile.MakeDirs(FLAGS.save_dir)
    
    if tf.gfile.Exists(FLAGS.save_vae_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_vae_dir)
    tf.gfile.MakeDirs(FLAGS.save_vae_dir)
    
    if tf.gfile.Exists(FLAGS.save_map_dir):
        tf.gfile.DeleteRecursively(FLAGS.save_map_dir)
    tf.gfile.MakeDirs(FLAGS.save_map_dir)

#%%
def get_data(filename):
    """ Load data from a xlsx file
    
    Parameters
    --------------------------------------------------------------------------
    file name:  string
                name of the .xlsx-file which data shall be loaded into the 
                workbench

    Returns
    --------------------------------------------------------------------------
    batchdata:  array (2D)
                returns a 2 dimensional array containing the data of the excel
                workbook
    """
    
    simulation_results = xlrd.open_workbook(filename)
    worksheet = simulation_results.sheet_by_index(0)
    
    batchdata = np.asarray(worksheet.col_values(0)).reshape((-1,1))
    
    num_col = worksheet.row_len(0)
    for i in range(1,num_col):
        batchdata = np.concatenate((batchdata, 
                np.asarray(worksheet.col_values(i)).reshape((-1,1))), axis=1)
    
    return batchdata


# load the input data
data = get_data('simulation_results.xlsx')
# the means and stds will be updated once K is inferenced
#data_mean_ctrl = data.mean(axis=0)
#data_std_ctrl = data.std(axis=0)
data_k = get_data('simulation_results_k.xlsx')
seqlen = len(data)
# divide input data into trainings and test set (70%/30%)  
seqlen_train = int(0.7*seqlen)
seqlen_test = seqlen - seqlen_train
n_samples = np.shape(data)[0]
parameters = np.shape(data)[1]

#%%
""" keep track of the neural network"""
# save for each assigned variable the mean, standard deviation, max value, 
# min value and the histogram
def variable_summaries(var):
    """ Creates summaries of tensors for TensorBoard visualization to keep 
    track of their development over time/iteration
    
    Parameters
    --------------------------------------------------------------------------
    var:  tensor
          weights or biases which shall be observed

    Returns
    --------------------------------------------------------------------------
    mean:      log file
               returns the mean value of the variable over time/iteration
    
    stddev:    log file
               returns the standard deviation of the variable over 
               time/iteration
           
    max:       log file
               returns the maximum value of the variable over time/iteration
           
    min:       log file
               returns the minimum value of the variable over time/iteration
           
    histogram: log file
               returns an histogram of the variable over time/iteration
    """
    
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def sech(x):
    # definition of the sech-function
    return np.cosh(x)**(-1)


def get(name):
    """ function to get a shared variabel given its name as a string
    
    Parameters
    --------------------------------------------------------------------------
    name:     string
              name of shared variabel

    Returns
    --------------------------------------------------------------------------
    tensor:    tensor
               returns a shared variable
    """
    tensor = tf.get_variable(name, dtype=tf.float32)
    
    return tensor

#%%
# dictionary for different training phases
train_step={}
loss ={}
loss_weights = {}
error = {}
max_error = {}
merged = {}
encoder_inp = {}
x_true = {}
decoder_inp = {}
decoder_outputs = {}
decoder_states = {}
keep_prob = {}
merge = {}
basic_cell = {}


#%%
"""
###############################################################################
##                  VAE to inference the latent variable K                   ##
###############################################################################
"""
np.random.seed(0)
tf.set_random_seed(0)
def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights
    
    Parameters
    --------------------------------------------------------------------------
    fan_in:   int
              number of inputs of the layer
              
    fan_out:  int
              number of outputs of the layer
              
    constant: float/int
              regularization of the maximum and minimum values

    Returns
    --------------------------------------------------------------------------
    distribution:
              array
              returns a random uniform distribution with the shape 
              [fan_in, fan_out] and maximum and minimum values
    """
    
    low = -constant*np.sqrt(6.0/(fan_in + fan_out)) 
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
    

class VariationalAutoencoder(object):
    """ Variation Autoencoder Class
    
    This implementation uses probabilistic encoders and decoders using Gaussian 
    distributions and  realized by multi-layer perceptrons. The VAE can be 
    learned end-to-end.
    
    It is a mapping of the feature inputs on a latent space and back to the 
    features itself. See "Auto-Encoding Variational Bayes" by Kingma and 
    Welling for more details.
    
    
    Parameters
    --------------------------------------------------------------------------
    feature input == feature outputs: 
        E:  tensor, shape = [None,1]
            energy function of the laser
        M4: tensor, shape = [None,1]
            fourth-moment (kurtosis) of the Fourier spectrum of the waveform
        αj: tensor, shape = [None,num_wave_plates]
            is a waveplate or polarizer angle (with j = 1, 2, 3, p).

    Returns
    --------------------------------------------------------------------------
    latent variable:
        z: tensor, shape = [None,1]
           representation of the birefringence

    Results
    --------------------------------------------------------------------------
    Sampeling from the latent space results in a representation for the
    birefringence. Since the birefringence cannot be measured, the 
    representation can be used instead.

    """
    
    def __init__(self, network_architecture, transfer_fct=tf.nn.relu, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        with tf.name_scope('input_placeholder'):
            self.x = tf.placeholder(tf.float32,
                                    [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        with tf.name_scope('Variational_Auto_encoder'):
            self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        self.latent_loss = tf.Variable(initial_value=[0,0])
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        #self.sess = tf.InteractiveSession()
        sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        with tf.name_scope('network_weights'):
            network_weights = self._initialize_weights(
                                    **self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_0, n_hidden_recog_1, 
                            n_hidden_recog_2, n_hidden_gener_0,
                            n_hidden_gener_1,  n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        with tf.name_scope('weights_recog'):
            all_weights['weights_recog'] = {
            'h0': tf.Variable(xavier_init(n_input, n_hidden_recog_0)),
            'h1': tf.Variable(xavier_init(n_hidden_recog_0, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        with tf.name_scope('biases_recog'):
            all_weights['biases_recog'] = {
            'b0': tf.Variable(tf.zeros([n_hidden_recog_0], dtype=tf.float32)),
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        with tf.name_scope('weights_gener'):
            all_weights['weights_gener'] = {
            'h0': tf.Variable(xavier_init(n_z, n_hidden_gener_0)),
            'h1': tf.Variable(xavier_init(n_hidden_gener_0, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        with tf.name_scope('biases_gener'):
            all_weights['biases_gener'] = {
            'b0': tf.Variable(tf.zeros([n_hidden_gener_0], dtype=tf.float32)),
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        with tf.name_scope('recog_layer_0'):
            layer_0 = self.transfer_fct(tf.add(
                    tf.matmul(self.x, weights['h0']), biases['b0'])) 
        with tf.name_scope('recog_layer_1'):
            layer_1 = self.transfer_fct(
                    tf.add(tf.matmul(layer_0, weights['h1']), biases['b1'])) 
        with tf.name_scope('recog_layer_2'):
            layer_2 = self.transfer_fct(
                    tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])) 
        with tf.name_scope('z_mean'):
            z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                                      biases['out_mean'])
        with tf.name_scope('z_log_sigma'):
            z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']), 
                             biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data 
        # space. The transformation is parametrized and can be learned.
        with tf.name_scope('generator_layer_0'):
            layer_0 = self.transfer_fct(tf.add(
                    tf.matmul(self.z, weights['h0']), biases['b0'])) 
        with tf.name_scope('generator_layer_1'):
            layer_1 = self.transfer_fct(tf.add(
                    tf.matmul(layer_0, weights['h1']), biases['b1'])) 
        with tf.name_scope('generator_layer_2'):
            layer_2 = self.transfer_fct(tf.add(
                    tf.matmul(layer_1, weights['h2']), biases['b2'])) 
        with tf.name_scope('reconstr_mean'):
            x_reconstr_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), 
                                               biases['out_mean'])
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        with tf.name_scope('reconstr_loss'):
            self.reconstr_loss = tf.nn.l2_loss(self.x - self.x_reconstr_mean)
        
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        #     between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        
        with tf.name_scope('latent_loss'):
            self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        
        # average over batch
        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)
        # Use ADAM optimizer
        self.optim = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.95, 
                                beta2=0.999, epsilon=1e-3)
        
        self.optimizer = self.optim.minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost, rec_loss, l_loss = sess.run((self.optimizer, self.cost,
                                        self.reconstr_loss, self.latent_loss), 
                                        feed_dict={self.x: X})
        return cost, rec_loss, l_loss
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})

#%% 

def VAE_train(network_architecture, learning_rate=0.0001,
          batch_size=100, training_epochs=10, display_step=5):
    """ Training Birefringence - Control Mapping.
    
    Parameters
    --------------------------------------------------------------------------
    feature input == feature outputs: 
        E:  tensor (1D)
            energy function of the laser
        M4: tensor (1D)
            fourth-moment (kurtosis) of the Fourier spectrum of the waveform
        αj: tensor (2D)
            is a waveplate or polarizer angle (with j = 1, 2, 3, p).
        
    
    Results
    --------------------------------------------------------------------------
    This function trains the Variational Autoencoder to be able to sample a
    representation of the birefringence from the latent space. Weights and 
    biases will be saved once the reconstruction is lower than a certain 
    threshold and the variation of the gradients decreases.
    
    """
    # Training cycle
    avg_cost = 100
    epoch = 0
    min_var = 100000
    #min_r_loss = 100000
    #for epoch in range(training_epochs):
    while(avg_cost > 0.1 and epoch < training_epochs):
        avg_cost = 0.
        avg_r_loss = 0.
        total_batch = int(n_samples / batch_size)
        np.random.shuffle(train_ind_VAE)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = np.reshape(data[int(train_ind_VAE[i]):
                int(train_ind_VAE[i]) + batch_size,:],[batch_size,parameters])
            if i == 0:
                print(int(train_ind_VAE[i]),int(train_ind_VAE[i]) + batch_size)

            # Fit training using batch data
            cost, r_loss, l_loss = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size
            avg_r_loss += r_loss / n_samples * batch_size
        
        # we want to have a function as smooth as possible to get a unique 
        # representation of the birefringence. Thus we look for latent space 
        # where the gradients have a small variance
        z_mu = vae.transform(data)
        z_norm = (z_mu - np.mean(z_mu,axis=0))/np.std(z_mu, axis=0)
        z_grad = np.gradient(z_norm, axis=0)
        z_std = np.std(z_grad,axis=0)
        z_ind = np.argmin(z_std)
        if (z_std[z_ind] < min_var and avg_r_loss < 30):
            min_var = min(z_std[0], z_std[1])
            print('min_var: %s' % min_var, avg_r_loss)
            saver_vae.save(sess, os.path.join(os.getcwd(),
                                              'Save_vae/trained_vars'),
                write_meta_graph=False, global_step=epoch+1)
                
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), 
                  "cost=", "{:.9f}".format(avg_cost),
                  "r_cost=", "{:.9f}".format(avg_r_loss))
            
        epoch += 1
        
    return vae

#%%
    
train_ind_VAE = np.linspace(0,len(data)-1-train_batch_size,
                            len(data)-train_batch_size)
np.random.shuffle(train_ind_VAE)

ys = tf.Variable(initial_value=np.linspace(0,len(data)-1,len(data)))

with tf.Session() as sess:
    # initialize the Variational Autoencoder (VAE)
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=0.0001,
                                 batch_size=train_batch_size)
    
    # intialize a safer to safe the VAE's weights and biases
    saver_vae = tf.train.Saver(max_to_keep=10)
    
    # initialize the test_writer of the VAE
    test_writer_vae = tf.summary.FileWriter(FLAGS.log_dir + '/test_vae')
    
    if not reuse_model:
        # in case we create a new model, the VAE will be trained
        vae = VAE_train(network_architecture,
                        training_epochs=FLAGS.max_steps_vae)
    
    # the best weights and biases will be restored
    saver_vae.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),
                                                               './Save_vae/')))
    
    # the representation of the birefringence will be sampled from the latent 
    # space. The representation is the normalized sample with a zero mean and a
    # standard deviation of 1. Since two latent variables were initialized, we 
    # choose the one which has the smaller standard deviation
    z_mu = vae.transform(data)
    z_mu_mean = np.mean(z_mu,axis=0)
    z_mu_std = np.std(z_mu, axis=0)
    z_norm = (z_mu - z_mu_mean)/z_mu_std
    z_grad = np.gradient(z_norm, axis=0)
    z_std = np.std(z_grad,axis=0)
    k_VAE_index = np.argmin(z_std)
    k_VAE = np.reshape(z_norm[:,k_VAE_index],[len(data),1])
    
    with open('k_VAE.csv', 'w',newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerows(k_VAE)
 
# extending the data set about the determined representation of the
# birefringence. This is the denormaized data set.
data = np.concatenate((data[:,:parameters-num_wave_plates],
                       k_VAE,data[:,parameters-num_wave_plates:]),axis=1)  

# keep the denormalized data set
d_dataset = data
# calculate the objective function for the data set
objective = d_dataset[:,0]/d_dataset[:,1]

# calculate mean, stddev, minimum and maximum for the data set
data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
data_min = data.min(axis=0)
data_max = data.max(axis=0)

# for learning the normalized data will be used
data = (data-data_mean)/data_std


#%%

def get_angles(k):
    """ Determining good angles within a certain interval around k.
    
    This implementation swips through the data set and searches for the best
    angles within a certain interval around the input k to maximize the 
    objective function.
    
    
    Parameters
    --------------------------------------------------------------------------
    k:  float
        representation of the birefringence

    Returns
    --------------------------------------------------------------------------
    ks:     float
            representation of the birefringence within the interval [k +- dx]
            where the objective function has the highest value
    output: array
            αj (with j = 1, 2, 3, p) from the data set for the birefringence 
            value ks
    obj:    float
            objective function value for the birefringence value ks

    Results
    --------------------------------------------------------------------------
    This function is used to generate a new data set. It is critical to
    determine a good data set for the birefringence-control mapping since very 
    small changes in the orientation of the angles will result in drastic 
    changes of the objective function. This function works as a filter 
    selecting only the birefringence representation and its angles which have
    the highest objective function values in a certain interval.

    """
    index = 0
    max_index, max_value = max(enumerate(objective),
                           key=operator.itemgetter(1))
    ks = k
    for t_step in range(1):
        obj = 0
        # find the best initial configuration for a given k in the
        # interval [0.98k, 1.02k]
        dx = (K_ub - K_lb)/50  #0.02
        for ind, item in enumerate(data[:,2]):
            #if (ind > 2*delay + 3):
                if (ks > K_lb + dx and ks < K_ub - dx):
                    if (item > ks-dx and item < ks+dx):
                        denorm_trainset = d_dataset[ind,:]
                        if obj < denorm_trainset[0]/denorm_trainset[1]:
                            obj = denorm_trainset[0]/denorm_trainset[1]
                            index = ind
                elif ks < K_lb + dx:
                    if (item >= K_lb and item < ks+dx):
                        denorm_trainset = d_dataset[ind,:]
                        if obj < denorm_trainset[0]/denorm_trainset[1]:
                            obj = denorm_trainset[0]/denorm_trainset[1]
                            index = ind
                elif ks > K_ub - dx:
                    if (item > ks-dx and item <= K_ub):
                        denorm_trainset = d_dataset[ind,:]
                        if obj < denorm_trainset[0]/denorm_trainset[1]:
                            obj = denorm_trainset[0]/denorm_trainset[1]
                            index = ind

        if obj == 0:
            obj = max_value

        if index > 0 and t_step == 0:
            max_index = index
            ks = data[max_index, 2]

        print(index, obj, ks)
        val = np.reshape(d_dataset[index,num_states:],[1,num_wave_plates])

    output = []
    for i in range(num_wave_plates):
        output.append(val[0,i])
    return ks,output,obj


#%%
# pre-learning of the birefringence
K_lb = min(d_dataset[:,2])
K_ub = max(d_dataset[:,2])

# define data set for K-u mapping
K_pre_steps = np.linspace(K_lb, K_ub, 1000)

# batch sizes for pretrainig and the actual control task
if (len(data)-2*delay-3-2*steps_phase_2 < 200):
    test_batch_1 = seqlen_test-2*(delay+steps_phase_1)-3
    test_batch_2 = seqlen_test-2*(delay+steps_phase_2)-3
else:
    test_batch_1 = 200
    test_batch_2 = 200

control_batch_size = 1

if reuse_model: #not
    K_u_list = []
    for i in range(len(K_pre_steps)):
        ks, output, obj = get_angles(K_pre_steps[i])
        val = np.concatenate(([ks], output, [abs(ks-K_pre_steps[i])], [obj]), 
                              axis=0)
        K_u_list.append(val)
    
    # sort the list, first by the birefringence value, than by the objective 
    # function value and thereafter by the difference between the input 
    # birefringence and the best birefrengence
    K_u_list = sorted(K_u_list, key=lambda x: (x[0], -x[-1], -x[-2]))
    K_u_list = np.vstack(K_u_list)
    i = 1
    length = len(K_u_list)
    # delete duplicates since only the best data points of the data set where 
    # choosen
    while i < length:
        if(K_u_list[i,0] == K_u_list[i-1,0]):
            K_u_list = np.delete(K_u_list,i-1,0)
            length -= 1
        else:
            i += 1
            
    K_u_list = np.delete(K_u_list,5,1)
    
    # calculate mean, stddev, and the gradients of the K_u_list
    K_u_mean = K_u_list.mean(axis=0)
    K_u_std = K_u_list.std(axis=0)
    K_u_mean[0] = 0
    K_u_std[0] = 1
    K_u = (K_u_list-K_u_mean)/K_u_std
    K_u_grad = np.gradient(K_u, axis=0)
    
    # if the gradient is small enough further data points will be included. 
    # This is advantageous to avoid overfitting.
    K_u_new = []
    for i in range(len(K_u)-1):
        K_u_new.append(K_u_list[i,:])
        max_val = np.max(abs(K_u_grad[i,:]))
        if max_val < 1:
            diff = K_u_list[i+1,:] - K_u_list[i,:]
            for m in range(1,1001):
                val = K_u_list[i,:] + m*diff/1000.0
                K_u_new.append(val)
    
    K_u_list = np.vstack(K_u_new)
    
    K_u_list = (K_u_list - K_u_mean)/K_u_std
    "specify data_mean and data_std that it won't get lost"
    #K_u_list[:,1:5] = (K_u_list[:,1:5]-data_mean[3:])/data_std[3:]
    
    # split K_u_list into training and test set     
    K_u_train = np.zeros([(len(K_u_list)//3)*2 + 1, num_wave_plates + 1])
    K_u_test = np.zeros([len(K_u_list)//3, num_wave_plates + 1])
    
    for step in range(len(K_u_list)//3):
        K_u_train[step*2:(step+1)*2] = K_u_list[step*3:2+step*3,:-1]
        K_u_test[step] = K_u_list[2+step*3,:-1]
        
    K_u_train[-1] = K_u_list[-1,:-1]
        
    train_ind = np.linspace(0,len(K_u_train)-101,len(K_u_train)-100)
    
    np.random.shuffle(train_ind)
    
else:
    K_u_test = np.zeros([len(K_pre_steps)//3,6])
    
batch_list = {'batch_1': train_batch_size,
              'batch_2': train_batch_size,
              'batch_11': test_batch_1,
              'batch_12': test_batch_2,
              'batch_3': control_batch_size,
              'batch_5': len(K_u_test)}

#%%
###############################################################################
##             Definition of the initail values of the weights               ##
###############################################################################

# necessary to identify the lstm trainables and to update them
lstm_variables = [None, None]

weight_list = [None]*6
weight_list[0] = np.zeros([num_parameters, hidden_layer_size],
                   dtype=np.float32)
weight_list[1] = np.zeros([delay*num_parameters, hidden_layer_size], 
                     dtype=np.float32)
weight_list[3] = np.zeros([num_wave_plates, hidden_layer_size],
                   dtype=np.float32)
weight_list[4] = np.zeros([delay*num_wave_plates, hidden_layer_size], 
                     dtype=np.float32)
 
for i in range(2):
    weight_list[i*3+2] = np.zeros([hidden_layer_size,], dtype=np.float32)


#%%

# The following trainables serve the purpose of identifying a mapping between
# the birefringence K and good control inputs
with tf.variable_scope('K-u_mapping'):
    with tf.name_scope('K-u_trainables'):
        map_weights = [tf.get_variable(
                name='map_weights_{}'.format(l),
                shape=[layer_struct[l],layer_struct[l+1]],
                initializer=tf.random_normal_initializer(0, 0.05),
                dtype=tf.float32) for l in range(len(layer_struct)-1)]
        map_biases = [tf.get_variable(
                name='map_biases_{}'.format(l),
                shape=[layer_struct[l+1],],
                initializer=tf.random_normal_initializer(0, 0.05),
                dtype=tf.float32) for l in range(len(layer_struct)-1)]
        for l in range(len(layer_struct)-1):
            variable_summaries(map_weights[l])
            variable_summaries(map_biases[l])

#%%
"""
###############################################################################
##                      Birefrigence - Control Mapping                       ##
###############################################################################
"""

# definition of fully connected layer
def new_fc_layer(input,          # The previous layer.
                 num_layer):     # Num. of layer.
    """ Creation of a new fully connected layer.
    
    Parameters
    --------------------------------------------------------------------------
    input:      tensor
                the previous layer of the deep NN
                
    num_layer:  float
                to characterize the layer and to select the shared weights and
                biases
                
    weights:    shared tensor
                shared weights which can be trained. The number of weights are 
                choosen depending on the NN structure defined through
                'layer_struct'
                
    biases:     shared tensor
                shared biases which can be trained. The number of biases are 
                choosen depending on the NN structure defined through
                'layer_struct'
    
    Returns
    --------------------------------------------------------------------------
    fc_layer:   tensor
                fully connected layer
    
    Results
    --------------------------------------------------------------------------
    This function creates a fully connected layer for a simple deep neural net.
    The network structure is given through 'layer_struct'.

    """                 
    # Create new weights and biases.
    weights = get('map_weights_' + str(num_layer))
    biases = get('map_biases_' + str(num_layer))

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    fc_layer = tf.matmul(input, weights) + biases
    
    if num_layer < (len(layer_struct)-2):
        fc_layer = tf.nn.relu(fc_layer)

    return fc_layer


def train_mapping(name, num=5, train=True):
    """ Train the mapping from the representation of the birefringence onto 
    good initial angles.
    
    Parameters
    --------------------------------------------------------------------------
    decoder_outputs['decoder_outputs_5']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        output of the neural network
        
    x_true['x_true_5']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        true output
                
    loss['loss_5']:
        tensorflow placeholder, shape = [None, ]
        L2 norm loss function defined as: output = sum(t ** 2) / 2
    
    error['error_5']:
        tensorflow placeholder, shape = [None, ]
        average error above the training/test set defined as: mean(l2_norm(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5')))
            
    max_error['max_error_5']:
        tensorflow placeholder, shape = [None, ]
        maximum error defined as: max(sum(square(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5'))))
    
    Results
    --------------------------------------------------------------------------
    This function creates the training procedure for the birefringence - 
    control mapping. The Adam algorithm is used for the optimization.

    """
    list_of_trainables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                scope='K-u_mapping')
    with tf.name_scope('loss_function'):
        decoder_outputs['decoder_outputs_' + str(num)] = \
        layer[-1]
        loss['loss_' + str(num)] = tf.reduce_mean(tf.nn.l2_loss(
                tf.subtract(decoder_outputs['decoder_outputs_' + str(num)],
                                             x_true.get('x_true_5'))))
        #max_error['max_error_' + str(num)] = 0.0
        max_error['max_error_' + str(num)] = tf.reduce_max(tf.reduce_sum(
            tf.square(tf.subtract(decoder_outputs['decoder_outputs_'+str(num)], 
                                  x_true.get('x_true_5'))),axis=1))
        tf.summary.scalar('loss_function', loss.get('loss_' + str(num)))
        
    with tf.name_scope('train_phase'):
        with tf.variable_scope(name):
            # Optimization algorithm:
            # AdagradOptimizer(FLAGS.learning_rate, name='AdaGrad')
            # RMSPropOptimizer(FLAGS.learning_rate,decay=0.9,
            #                  momentum=FLAGS.momentum,
            #                  name='RMSProp_control')
            # ...
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                               beta1=0.95, beta2=0.999,
                                               epsilon=1e-3)
            train_step['train_step_' + str(num)] = optimizer.minimize(
                    loss.get('loss_' + str(num)),
                    var_list=list_of_trainables)
    with tf.name_scope('error'):
        error['error_' + str(num)] = \
        loss.get('loss_' + str(num))/batch_list.get('batch_' + str(num))
        tf.summary.scalar('error', error.get('error_' + str(num)))
            
        # Merge all the summaries and write them out to
        merged['merged_' + str(num)] = tf.summary.merge(
            [tf.summary.scalar('error',error.get('error_' + str(num))),
             tf.summary.scalar('max_error',max_error.get('max_error_' + str(num))),
             tf.summary.scalar('loss_function',loss.get('loss_' + str(num)))])
        #merge_all

with tf.name_scope('Birefringence-control_mapping'):
    # definition of the Birefrigence input placeholder K
    K = tf.placeholder(shape=[None,1],dtype=tf.float32, name='K_value')
    x_true['x_true_5'] = tf.placeholder(shape=[None,num_wave_plates],
                                        dtype=tf.float32, name='true_ctrl')
    # in order to prevent the deep MPC from divergating after a couple of 
    # iterations, the angles are clipped to the minimum and maximum value of 
    # the data set
    angles_min = tf.Variable(initial_value=data_min[num_states:],
                             trainable=False, dtype=tf.float32)
    angles_max = tf.Variable(initial_value=data_max[num_states:],
                             trainable=False, dtype=tf.float32)
    #decoder_outputs['decoder_outputs_5'] = tf.placeholder(
    #                           shape=[None,num_wave_plates], dtype=tf.float32,
    #                           name='dec_out_5')
    layer = [None]*(len(layer_struct)-1)
    with tf.variable_scope('K-u_mapping', reuse=True):
        with tf.name_scope('layer_0'):
            layer[0] = new_fc_layer(input=K, num_layer=0)
        for num in range(1,len(layer_struct)-1):
            with tf.name_scope('layer_' + str(num)):
                layer[num] = new_fc_layer(input=layer[num-1], num_layer=num)
                if num == len(layer_struct)-2:
                    # clipping output of the K-u mapping by value to avoid
                    # divergating
                    tf.clip_by_value(layer[num],angles_min, angles_max)
                
    train_mapping('K-u_mapping')

#%%
"""
###############################################################################
##                     Birefringence - Control Mapping                       ##
###############################################################################
"""

def K_u_mapping():
    """ Training Birefringence - Control Mapping.
    
    Parameters
    --------------------------------------------------------------------------
    decoder_outputs['decoder_outputs_']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        output of the neural network
        
    x_true['x_true_5']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        true output
                
    loss['loss_5']:
        tensorflow placeholder, shape = [None, ]
        L2 norm loss function defined as: output = sum(t ** 2) / 2
    
    error['error_5']:
        tensorflow placeholder, shape = [None, ]
        average error above the training/test set defined as: mean(l2_norm(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5')))
            
    max_error['max_error_5']:
        tensorflow placeholder, shape = [None, ]
        maximum error defined as: max(sum(square(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5'))))
    
    Results
    --------------------------------------------------------------------------
    This function trains the Birefringence-control mapping. The data set is 
    splitted into a training and a test set. Weights and biases are saved once 
    average error of the test set further decreased.
    
    """
    count = 0
    print('Start Bifringence control mapping...')
    epoch = 0
    #test_err = 1000
    
    err_temp = 100000
    #max_err_temp = 100000
    
    while (err_temp > 1e-6 and epoch < FLAGS.max_steps_K_u):
        #temp_err = test_err

        if count == len(K_u_train)-100:
            np.random.shuffle(train_ind)
            count = 0
        if (epoch) % 100 == 0:  # Record summaries and test-set accuracy
            feed_dict = {K: np.reshape(K_u_test[:len(K_u_test),0], [len(K_u_test),1]),
                         x_true['x_true_5']: K_u_test[:len(K_u_test),1:]}
            summary, err_testset, max_err = sess.run([merged.get('merged_5'),
                                     error.get('error_5'),
                                     max_error.get('max_error_5')],
                                     feed_dict=feed_dict)
            #test_err = err_testset
            test_writer_map.add_summary(summary, epoch)
            print('Error in iteration %s: %s %s' % (epoch,err_testset,max_err))
            

            #a = np.array((err_temp,max_err_temp))
            #b = np.array((err_testset,max_err))
            #rel = sum((a-b)/np.array((max(a[0],b[0]),max(a[1],b[1]))))
            #if rel > 0:
            if err_testset < err_temp:
                err_temp = err_testset
                #max_err_temp = max_err
                
                print('+++ Save weights and biases+++')
                saver_map.save(sess, os.path.join(os.getcwd(),'Save_map/trained_vars'),
                           write_meta_graph=False, global_step=epoch)
        else:
            if (epoch) % 100 == 9:  # Record execution stats
                feed_dict = {K: np.reshape(K_u_train[int(
                        train_ind[count]):int(train_ind[count])+100,0],[100,1]),
                             x_true['x_true_5']: np.reshape(
                               K_u_train[int(
                               train_ind[count]):int(train_ind[count])+100,1:],
                               [100,num_wave_plates])}
                run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                summary,_ = sess.run([merged.get('merged_5'),
                                      train_step.get('train_step_5')],
                                      feed_dict=feed_dict,
                                      options=run_options)
                train_writer_map.add_summary(summary, epoch)
            else:  # Record a summary
                feed_dict = {K: np.reshape(K_u_train[int(
                        train_ind[count]):int(train_ind[count])+100,0],[100,1]),
                             x_true['x_true_5']: np.reshape(
                               K_u_train[int(
                               train_ind[count]):int(train_ind[count])+100,1:],
                               [100,num_wave_plates])}
                sess.run([merged.get('merged_5'),
                          train_step.get('train_step_5')],
                          feed_dict=feed_dict)
        
        epoch +=1
        count += 1


#%%
# K_u_list contains the data points for learning the mapping from the
# birefringence to the control inputs

with tf.Session() as sess:
    saver_map = tf.train.Saver(max_to_keep=10)

    sess.run(tf.global_variables_initializer())
            
    train_writer_map = tf.summary.FileWriter(FLAGS.log_dir + '/train_map')
    test_writer_map = tf.summary.FileWriter(FLAGS.log_dir + '/test_map')
    
    if not reuse_model:
        K_u_mapping()
            
    saver_map.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),
                                                      './Save_map/')))
    #map_out = sess.run([layer[-1]],feed_dict={K: np.reshape(K_u_list[:,0],
    #                                                      [len(K_u_list),1])})
    #map_out = map_out[0]
    
    map_out_2 = sess.run([layer[-1]],feed_dict={K: k_VAE})
    map_out_2 = map_out_2[0]
    
    map_o = map_out_2*K_u_std[1:5]+K_u_mean[1:5]
    #map_o2 = map_out_2*data_std[3:] + data_mean[3:]
    
    #inputs = np.concatenate((np.reshape(data_k[:,2],[len(data_k),1]),
    #                                                 map_o),axis=1)
    
    #with open('inputs.csv', 'w',newline='') as csvfile:
    #    spamwriter = csv.writer(csvfile)
    #    for row in inputs:
    #        spamwriter.writerow(row)
    
    """
    
    """
    if new_dataset:
        # if no new data set was created yet, it will be created here.
        # For the former data set, the angles were choosen randomly, i.e. not
        # according to the given birefringence. After the K-u mapping, a data 
        # set can be created where there is a mapping from the birefringence to
        # the angles. This is in particular useful for the control part when 
        # predicting several steps into the future, since the states of the 
        # system depend on the angles and will be feed back into the MPC.
        map_out = sess.run([layer[-1]],feed_dict={K: k_VAE})
                
        map_out = map_out[0]
            
        s = []
        T = 60
        n = 256
        t2 = np.linspace(-T/2,T/2,n+1)
        t_dis = t2[0:n].reshape([1,n])      # time discretization
        u=np.reshape(sech(t_dis/2), [n,])   # orthogonally polarized electric field 
        v=np.reshape(sech(t_dis/2), [n,])   # envelopes in the optical fiber
        ut=np.fft.fft(u).reshape(n,)        # fast fourier transformation of the
        vt=np.fft.fft(v).reshape(n,)        # electrical fields
        uvt=np.concatenate([ut, vt], axis=0)# concatenation of ut and vt
        for i in range(len(map_out)):
            if i % 100 == 0:
                print(i)
            (_,states) = laser_simulation(uvt, map_out[i,0], map_out[i,1], 
            map_out[i,2], map_out[i,3], data_k[i,2])
            s.append(states)
                    
        s = np.vstack(s)
        
        with open('simulation_results_new_angles.csv', 'w',newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for row in s:
                spamwriter.writerow(row)
                
        data = np.concatenate((s[:,:parameters-num_wave_plates],
                               k_VAE,s[:,parameters-num_wave_plates:]),axis=1) 
         
        with open('simulation_results_new_angles_k.csv', 'w',newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for row in data:
                spamwriter.writerow(row)
                
    # load the input data
    # simulation_results__new_angles_k.xlsx
    data_no_K = get_data('simulation_results_new_angles.xlsx')
    # the means and stds will be updated once K is inferenced
    data_mean_ctrl = data_no_K.mean(axis=0)
    data_std_ctrl = data_no_K.std(axis=0)
    data_k = get_data('simulation_results_new_angles_k.xlsx')
    seqlen = len(data_no_K)
    # divide input data into trainings and test set (70%/30%)  
    seqlen_train = int(0.7*seqlen)
    seqlen_test = seqlen - seqlen_train
    parameters = np.shape(data_k)[1]
    
    data = np.concatenate((data_no_K[:,:parameters-num_wave_plates-num_latent_var],
                           k_VAE,data_no_K[:,parameters-num_wave_plates-num_latent_var:]),axis=1)
    
    
    # define the denormalized data set
    d_dataset = data
    # calculate the objective function values of the new data set
    objective = d_dataset[:,0]/d_dataset[:,1]
    
    # calculate the mean, stddev, maximum and minimum of the new data set not
    # containing the birefringence K
    data_no_K_mean = data_no_K.mean(axis=0)
    data_no_K_std = data_no_K.std(axis=0)
    data_no_K_max = data_no_K.max(axis=0)
    data_no_K_min = data_no_K.min(axis=0)
    
    # normalization of the data set not containing the birefringence K
    data_no_K = (data_no_K-data_no_K_mean)/data_no_K_std
    
    # spliting data set into training and test set
    train_data = data_no_K[0:seqlen_train,:]
    test_data = data_no_K[seqlen_train:seqlen,:]
    
    # calculate the mean, stddev, maximum and minimum of the new data set
    data_mean = data.mean(axis=0)
    data_mean[2] = 0
    data_std = data.std(axis=0)
    data_std[2] = 1
    data_max = data.max(axis=0)
    data_min = data.min(axis=0)
    
    # normalization of the data set
    #data = (data-data_mean)/data_std
    
    # spliting data set into training and test set. This is needed for the learning
    # task, since the future value of the birefringence will be predicted.
    data_norm_K = 1 + 9*(d_dataset-data_min)/(data_max-data_min)
    train_data_K = data_norm_K[0:seqlen_train,:]
    test_data_K = data_norm_K[seqlen_train:seqlen,:]
    
    data_min32 = np.array(data_no_K.min(axis=0), dtype=np.float32)
    data_max32 = np.array(data_no_K.max(axis=0), dtype=np.float32)
    sess.run(angles_min.assign(data_min32[num_states-num_latent_var:]))
    sess.run(angles_max.assign(data_max32[num_states-num_latent_var:]))
    
    tf.clip_by_value(layer[-1],angles_min, angles_max)



x = np.linspace(0,7,30000)
K_sim = -.075 + .175*np.sin(0.5*x**2)
K_steps = (K_sim-K_sim.mean())/K_sim.std()

#%%

if not reuse_model:
    """
    ###########################################################################
    ##                         Pre-Training Phase 1                          ##
    ###########################################################################
    """  
    # Using Conditional Restricted Boltzman Machines (CRBM) i order to minimize 
    # the energy between input and hidden layer.
    # This is the first pre-learning step in order to get better initial values 
    # for the initial weights
    
    # Data list for pre-training
    data_list = [data_no_K, data[:,num_states:]]
    """ pre-training using Conditional Restricted Boltzmann Machines (CRBM)"""
    with tf.name_scope('pre_training'):
        weight_list=[]
        for dataset in data_list:
            # pre trianing using Conditional Restrictes Boltzman Machines (CRBM)
            crbm = train_crbm(dataset=dataset, n_hidden=hidden_layer_size,
                              training_epochs=FLAGS.max_pre_steps, delay=delay)
            # saving pre learned weights in a list -> will be used for the 
            # second pre learning step
            weight_list.append(np.float32(crbm.W.eval()))
            weight_list.append(np.float32(crbm.B.eval()))
            weight_list.append(np.float32(crbm.hbias.eval()))

#%%
"""
###############################################################################
##                              Initialization                               ##
###############################################################################
"""
    
       
###############################################################################
##                      Initialization of trainables                         ##
###############################################################################
    
# Initializtion of the trainables (weights and biases) which were mentioned 
# before. These trainables will be used within the RNN cell. That is why it is 
# necessary to set the variable scope to this specific name
    
# The following trainables serve the purpose of MPC
def var_initialization():
    with tf.variable_scope('combined_tied_rnn_seq2seq/tied_rnn_seq2seq/RNN_parameters'):
        with tf.name_scope('hidden_layer_lc'):
            with tf.name_scope('current_weigths'):
                W_lc_current = tf.get_variable(name='W_lc_current',
                                               initializer=weight_list[0],
                                               dtype=tf.float32)
                variable_summaries(W_lc_current)
            with tf.name_scope('history_weights'):
                W_lc_past = tf.get_variable(name='W_lc_past',
                                            initializer=weight_list[1],
                                            dtype=tf.float32)
                variable_summaries(W_lc_past)
            with tf.name_scope('biases'):    
                B_lc = tf.get_variable(name='B_lc',
                                       initializer=weight_list[2],
                                       dtype=tf.float32)
                variable_summaries(B_lc)
        with tf.name_scope('hidden_layer_lp'):
            with tf.name_scope('current_weigths'):
                W_lp_current = tf.get_variable(name='W_lp_current',
                                               initializer=weight_list[0],
                                               dtype=tf.float32)
                variable_summaries(W_lp_current)
            with tf.name_scope('history_weights'):
                W_lp_past = tf.get_variable(name='W_lp_past',
                                            initializer=weight_list[1],
                                            dtype=tf.float32)
                variable_summaries(W_lp_past)
            with tf.name_scope('biases'):    
                B_lp = tf.get_variable(name='B_lp',
                                       initializer=weight_list[2],
                                       dtype=tf.float32)
                variable_summaries(B_lp)
        with tf.name_scope('hidden_layer_c'):    
            with tf.name_scope('current_weigths'):
                W_c_current = tf.get_variable(name='W_c_current',
                                              initializer=weight_list[0],
                                              dtype=tf.float32)
                variable_summaries(W_c_current)
            with tf.name_scope('history_weights'):
                W_c_past = tf.get_variable(name='W_c_past',
                                           initializer=weight_list[1],
                                           dtype=tf.float32)
                variable_summaries(W_c_past)
            with tf.name_scope('biases'):    
                B_c = tf.get_variable(name='B_c',
                                      initializer=weight_list[2],
                                      dtype=tf.float32)
                variable_summaries(B_c)
        with tf.name_scope('hidden_layer_f'):    
            with tf.name_scope('current_weigths'):
                W_f_current = tf.get_variable(name='W_f_current',
                                              initializer=weight_list[3],
                                              dtype=tf.float32)
                variable_summaries(W_f_current)
            with tf.name_scope('history_weights'):
                W_f_past = tf.get_variable(name='W_f_past',
                                           initializer=weight_list[4],
                                           dtype=tf.float32)
                variable_summaries(W_f_past)
            with tf.name_scope('biases'):    
                B_f = tf.get_variable(name='B_f',
                                      initializer=weight_list[5],
                                      dtype=tf.float32)
                variable_summaries(B_f)
        with tf.name_scope('latent_layer_l'):    
            with tf.name_scope('current_weigths'):
                W_l = tf.get_variable(name='W_l',
                            shape=[latent_layer_size,hidden_layer_size],
                            initializer=tf.random_normal_initializer(0, 0.05),
                            dtype=tf.float32)
                variable_summaries(W_l)
            with tf.name_scope('biases'):    
                B_l = tf.get_variable(name='B_l', shape=[hidden_layer_size,],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
                variable_summaries(B_l)
        with tf.name_scope('hidden_layer_l'):
            with tf.name_scope('current_weigths'):
                W_lo = tf.get_variable(name='W_lo',
                            initializer=xavier_init(hidden_layer_size,
                                                    latent_layer_size),
                            dtype=tf.float32)
                variable_summaries(W_lo)
            with tf.name_scope('biases'):    
                B_lo = tf.get_variable(name='B_lo', shape=[latent_layer_size,],
                                   initializer=tf.constant_initializer(0.0),
                                   dtype=tf.float32)
                variable_summaries(B_lo)
        with tf.name_scope('output_layer'):
            with tf.name_scope('current_weigths'):
                W_o = tf.get_variable(name='W_o',
                            initializer=xavier_init(hidden_layer_size,
                                                    num_states-num_latent_var),
                            dtype=tf.float32)
                variable_summaries(W_o)
                W_ol = tf.get_variable(name='W_ol',
                            initializer=xavier_init(hidden_layer_size,
                                                    num_latent_var),
                            dtype=tf.float32)
                variable_summaries(W_ol)
            with tf.name_scope('biases'):    
                B_o = tf.get_variable(name='B_o', shape=[num_states-num_latent_var,],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
                variable_summaries(B_o)
                B_ol = tf.get_variable(name='B_ol', shape=[num_latent_var,],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
                variable_summaries(B_ol)
        with tf.name_scope('latent_layer_ll'):
            with tf.name_scope('current_weigths'):
                W_ll = tf.get_variable(name='W_ll',
                            initializer=xavier_init(latent_layer_size,
                                                    latent_layer_size),
                            dtype=tf.float32)
                variable_summaries(W_ll)
        with tf.name_scope('next_hidden_layer'):    
            with tf.name_scope('hidden_weigths'):
                W_h = [tf.get_variable(name='W_h_{}'.format(t),
                            initializer=xavier_init(hidden_layer_size,
                                                    hidden_layer_size),
                            dtype=tf.float32) for t in range((layers-1)*delay)]
                for t in range((layers-1)*delay):
                    variable_summaries(W_h[t])
            with tf.name_scope('hidden_biases'):
                B_h = [tf.get_variable(name='B_h_{}'.format(t),
                            shape=[hidden_layer_size,],
                            initializer=tf.random_normal_initializer(0, 0.0),
                            dtype=tf.float32) for t in range((layers-1)*delay)]
                for t in range((layers-1)*5):
                    variable_summaries(B_h[t])
                
    with tf.variable_scope('combined_tied_rnn_seq2seq/tied_rnn_seq2seq/myRNNCell/RNN_parameters'):
        with tf.name_scope('control_input'):
            objective = d_dataset[:,0]/d_dataset[:,1]
            max_index, max_value = max(enumerate(objective),
                                       key=operator.itemgetter(1))
            control_input = [tf.get_variable(
                                name='control_input_{}'.format(t),
                                initializer=np.float32(
                                    d_dataset[max_index,num_states:]).reshape(
                                            [control_batch_size,
                                             num_wave_plates]),
                          dtype=tf.float32) for t in range(steps_phase_crtl)]
            for t in range(steps_phase_crtl):
                variable_summaries(control_input[t])
                
    return (W_lc_current, W_lc_past, B_lc, W_lp_current, W_lp_past, B_lp, W_c_current, 
    W_c_past, B_c, W_f_current, W_f_past, B_f, W_l, B_l, W_lo, B_lo, W_o, W_ol, 
    B_o, B_ol, W_ll, W_h, B_h, control_input)


#%%
# definition of the LSTM cell used to capture long term dynamics of the
# dynamical system
latent_cell = tf.contrib.rnn.LSTMCell(latent_layer_size)   
  

(W_lc_current, W_lc_past, B_lc, W_lp_current, W_lp_past, B_lp, W_c_current, 
W_c_past, B_c, W_f_current, W_f_past, B_f, W_l, B_l, W_lo, B_lo, W_o, W_ol, 
B_o, B_ol, W_ll, W_h, B_h, control_input) = var_initialization()
    
# merge all the summaries of the trainables
# These are the basic trainable which are updated in each pre training step.
# There are further trainables for each specific phase that cannot be listed
# here. They have to be added at the specific position.
merge_all = tf.summary.merge_all()


#%%
# Create a RNN cell using the NN which was defined before
# The structure of this class must be the same as pre defined by tensorflow for
# RNN. This includes __init__, state_size, output_size, __call__
obs = []
obs_p_past = []
obs_p_cur = []
obs_c_past = []
obs_c_cur = []
obs_f_past = []
obs_f_cur = []

obs_1 = []
obs_p_past_1 = []
obs_p_cur_1 = []
obs_c_past_1 = []
obs_c_cur_1 = []
obs_f_past_1 = []
obs_f_cur_1 = []

obs_act_lo = []
obs_act_l = []
obs_act_l1 = []

observation_dec = []

class myRNNCell(RNNCell):
  """ Create own RNNCell """

  def __init__(self, num_units, state_is_tuple=False, control=False,
               lstm_reuse=False, batch_size=100, time_steps=1):
    """
    Args:
        num_units: int, The output size of the RNN cell (Energy and kurtosis).
        state_is_tuple: If True, accepted and returned states are 2-tuples of
          the `c_state` and `m_state`.  By default (False), they are 
          concatenated along the column axis.  This default behavior will soon
          be deprecated.
        conrtol: bool, If True, the control task will be treated
        lstm_reuse: bool, If False, new LSTM trainables will be created, If
          True these trainables will be reused
        batch_size: int, Defines the batch size of the input.
    """
    if not state_is_tuple:
        logging.warn("%s: Using a concatenated state is slower and will soon be"
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._state_is_tuple = state_is_tuple
    self._control = control
    self._lstm_reuse = lstm_reuse
    self._batch_size = batch_size
    self._time_steps = time_steps
    
  @property
  def state_size(self):
      return (self._num_units, latent_layer_size)

  @property
  def output_size(self):
      return self._num_units

  def __call__(self, inputs, state, scope=None):
      """ Definition of the RNN 
      Args:
          inputs: tensor, Will be feed in using the feed dictionary. The size
            is [2*time_steps, self._batch_size, num_inputs_RNN]
          state: tuple, Includes c and h of the previous step, where c is the 
            state of the laser [Energy, Kurtosis], and h is the output of the 
            latent layer of the previous cell
      """
      
      print(inputs.name, inputs.shape)
      
      # initialize the zero state of the LSTM Cell
      #initial_state_latent = latent_cell.zero_state(self._batch_size,
      #                                              dtype=tf.float32)

      # get the shared variables (trainables of the RNN cell)
      with tf.variable_scope('RNN_parameters', reuse=True):
              W_lp_current = get('W_lp_current')
              W_lp_past = get('W_lp_past')
              B_lp = get('B_lp')
              W_lc_current = get('W_lc_current')
              W_lc_past = get('W_lc_past')
              B_lc = get('B_lc')
              W_c_current = get('W_c_current')
              W_c_past = get('W_c_past')
              B_c = get('B_c')
              W_f_current = get('W_f_current')
              W_f_past = get('W_f_past')
              B_f = get('B_f')
              W_l = get('W_l')
              B_l = get('B_l')
              W_lo = get('W_lo')
              B_lo = get('B_lo')
              W_o = get('W_o')
              B_o = get('B_o')
              W_ol = get('W_ol')
              B_ol = get('B_ol')
              W_ll = get('W_ll')
              for i in range(delay):
                  W_h.append(get('W_h_%s' % i))
                  B_h.append(get('B_h_%s' % i))
      
      with vs.variable_scope(scope or type(self).__name__):
          # get the output of the previous cell (output of the previous cell 
          # 'c' and and output of the previous latent layer 'h')
          if self._state_is_tuple:
              c, h = state
          else:
              c, h = array_ops.split(1, 2, state)
              
          # in case it is not the control phase:
          #    Division of the inputs:
          #    inp_p_past[t-11, t-10, t-9, t-8, t-7]
          #    inp_p_cur[t-6]
          #    inp_c_past[t-5, t-4, t-3, t-2, t-1]
          #    inp_c_cur[t]
          #    inp_f_past[t-4, t-3, t-2, t-1, t]
          #    inp_f_cur[t+1]
          #if not self._control or threshold==0:
          global threshold,lstm_variables
          #print(threshold)
          #if ('GO' not in inputs.name or threshold == 0):
          if ('GO' not in inputs.name or threshold == 0 or not self._control):
              """
              if self._control:
                  split_1,split_2,split_3 = tf.split(c, num_or_size_splits=3, 
                                                     axis=1)
                  split_3 = tf.scalar_mul(-1.0/data_std[2],
                                          tf.subtract(mean_K, split_3))
                  c_dict[threshold] = tf.concat((split_1,split_2,split_3),
                        axis=1)
              """
              
              with tf.name_scope('input_cell'):
                  inp_p_past = tf.slice(inputs,[0,0],[self._batch_size,
                                        num_parameters*delay])
                  inp_p_cur = tf.slice(inputs,[0,delay*num_parameters],
                                       [self._batch_size,num_parameters])
                  inp_c_past = tf.slice(inputs,[0,(delay+1)*num_parameters],
                                    [self._batch_size,num_parameters*delay])
                  inp_c_cur = tf.slice(inputs,[0,(2*delay+1)*num_parameters],
                                    [self._batch_size,num_parameters])
                  inp_f_past = tf.slice(inputs,[0,(2*delay+2)*num_parameters],
                                    [self._batch_size,num_wave_plates*delay])
                  inp_f_cur = tf.slice(inputs,
                          [0,(delay+1)*2*num_parameters+delay*num_wave_plates],
                          [self._batch_size,num_wave_plates])
                  
          # in case it is the control phase:
          #    Division of the inputs:
          #    inp_p_past[t-11, t-10, t-9, t-8, t-7]
          #    inp_p_cur[t-6]
          #    inp_c_past[t-5, t-4, t-3, t-2, t-1]
          #    inp_c_cur[t]
          #    inp_f_past[t-4, t-3, t-2, t-1, t]
          #    inp_f_cur[t+1]
          # Here we feed the new output plus the determined angles back as 
          # input for the next step
          #if self._control:
          #if ('GO' in inputs.name or threshold > 0):
          if (('GO' in inputs.name or threshold > 0) and self._control):
              split_1,split_2,split_3 = tf.split(c, num_or_size_splits=3, 
                                                axis=1)
              split_1 = tf.add(tf.multiply(tf.divide(tf.subtract(
                      split_1, one),nine), tf.subtract(ener_max, ener_min)),
                      ener_min)
              split_1 = tf.scalar_mul(-1.0/data_std[0],
                                      tf.subtract(mean_Energy, split_1))
              
              split_2 = tf.add(tf.multiply(tf.divide(tf.subtract(
                      split_2, one),nine), tf.subtract(moment_max, 
                      moment_min)), moment_min)
              split_2 = tf.scalar_mul(-1.0/data_std[1],
                                      tf.subtract(mean_Kurtosis, split_2))
              
              #split_3 = tf.scalar_mul(-1.0/data_std[2],
              #                        tf.subtract(mean_K, split_3))
              c_dict[threshold] = tf.concat((split_1,split_2),
                    axis=1)
              with tf.name_scope('input_cell'):
                  if delay + 3 > threshold:
                      inp_p_past = tf.slice(inputs,[0,0],[self._batch_size,
                                            num_parameters*delay])
                  else:
                      inp_p_past = tf.slice(inputs,[0,0],[self._batch_size,
                                            num_parameters*(
                                            2*delay+2-threshold)])
                  if delay + 2 > threshold:
                      inp_p_cur = tf.slice(inputs,
                                           [0,delay*num_parameters],
                                           [self._batch_size,
                                            num_parameters])
                      
                  if threshold == 0:
                      inp_c_past = tf.slice(inputs,
                                            [0,(delay+1)*num_parameters],
                                            [self._batch_size,
                                             num_parameters*delay])
                      inp_c_cur = tf.slice(inputs,
                                           [0,(2*delay+1)*num_parameters],
                                           [self._batch_size,
                                            num_parameters])
                  
                  if delay + 1 > threshold and threshold > 0:
                      inp_c_past = tf.slice(inputs,
                                      [0,(delay+1)*num_parameters],
                                      [self._batch_size, num_parameters*(
                                       delay-threshold+1)])
              if (self._time_steps == steps_phase_crtl
                  and self._batch_size == 1):
                  obs_p_past.append(inp_p_past)
                  obs_p_cur.append(inp_p_cur)
                  obs_c_past.append(inp_c_past)
                  obs_c_cur.append(inp_c_cur)
                  obs_f_past.append(inp_f_past)
                  obs_f_cur.append(inp_f_cur)
                  
              if self._control:
                   # if self._control, the control inputs will be trainables
                   # which can be updated in order to optimize the objective 
                   # function. Since we predict several time steps into the
                   # future, we feed the output of time step t back as input 
                   # for time step t+1
                   with tf.variable_scope('RNN_parameters', reuse=True):
                      if delay > threshold:
                          inp_f_past = tf.slice(inputs,
                                      [0,12*num_parameters],[self._batch_size,
                                      num_wave_plates*(delay-threshold)])
                      if delay + 2 < threshold:
                          for t in range(threshold-delay-2):
                              angle = tf.concat((c_dict[t+1],
                                tf.get_variable('control_input_{}'.format(t), 
                                                     dtype=tf.float32)),axis=1)
                              inp_p_past = tf.concat((inp_p_past,angle),axis=1)                      
                      
                      if delay + 1 < threshold:
                          inp_p_cur = tf.concat((c_dict[threshold-delay-1], 
                                      tf.get_variable(
                                              'control_input_{}'.format(
                                              threshold-delay-2), 
                                              dtype=tf.float32)),axis=1)
        						  
                      if threshold > 0:
                          inp_c_cur = tf.concat((c_dict[threshold],
                                                 tf.get_variable(
                                        'control_input_{}'.format(threshold-1), 
                                        dtype=tf.float32)),axis=1)
                      if delay > threshold:
                          for t in range(threshold):
                              inp_f_past = tf.concat((
                                inp_f_past,
                                tf.get_variable('control_input_{}'.format(t), 
                                dtype=tf.float32)),axis=1)
                      else:
                          for t in range(threshold-delay,threshold):
                              if t == threshold-delay:
                                  inp_f_past = tf.get_variable(
                                                  'control_input_{}'.format(t), 
                                                  dtype=tf.float32)
                              else:
                                  inp_f_past = tf.concat((
                                    inp_f_past,
                                    tf.get_variable(
                                        'control_input_{}'.format(t), 
                                        dtype=tf.float32)),axis=1)
                                  
                      if delay+1 > threshold and threshold > 0:
                          for t in range(threshold-1):
                              angle = tf.concat((c_dict[t+1],
                                tf.get_variable('control_input_{}'.format(t), 
                                dtype=tf.float32)),axis=1)
                              inp_c_past = tf.concat((
                                inp_c_past,angle),axis=1)
                              
                      elif delay < threshold:
                          for t in range(threshold-delay-1,threshold-1):
                              if t == threshold-delay-1:
                                      inp_c_past = tf.concat((c_dict[t+1],
                                      tf.get_variable(
                                              'control_input_{}'.format(t), 
                                              dtype=tf.float32)),axis=1)
                              else:
                                  angle = tf.concat((c_dict[t+1],
                                        tf.get_variable(
                                            'control_input_{}'.format(t), 
                                            dtype=tf.float32)),axis=1)
                                  inp_c_past = tf.concat((
                                        inp_c_past,angle),axis=1)

                      inp_f_cur = tf.get_variable(
                                          'control_input_{}'.format(threshold), 
                                          dtype=tf.float32)
                      
                      if (self._time_steps == steps_phase_crtl 
                         and self._batch_size == 1):
                          obs_p_past_1.append(inp_p_past)
                          obs_p_cur_1.append(inp_p_cur)
                          obs_c_past_1.append(inp_c_past)
                          obs_c_cur_1.append(inp_c_cur)
                          obs_f_past_1.append(inp_f_past)
                          obs_f_cur_1.append(inp_f_cur)
                        
              else:
                  # if it is not the control task, the predicted states will be
                  # feed back as input for the next state, but the angles will
                  # be used from the data set
                  inp_f_past = tf.slice(inputs,[0,(2*delay+2)*num_parameters],
                                  [self._batch_size,num_wave_plates*delay])
                  if delay + 2 < threshold:
                      for t in range(threshold-delay-2):
                          angle = tf.concat((c_dict[t+1],
                                  tf.slice(inputs,[0, (delay - 2 + t) * \
                                  num_parameters + num_states - num_latent_var],
                                  [self._batch_size,num_wave_plates])),
                                  axis=1)
                          inp_p_past = tf.concat((inp_p_past,angle),axis=1)                      
                      
                  if delay + 1 < threshold:
                      inp_p_cur = tf.concat((c_dict[threshold-delay-1],
                                  tf.slice(inputs,[0,
                                  delay * num_parameters + num_states - num_latent_var],
                                  [self._batch_size,num_wave_plates])),
                                  axis=1)
        						  
                  if threshold > 0:
                      inp_c_cur = tf.concat((c_dict[threshold],
                                  tf.slice(inputs,[0, (2*delay+1) * \
                                           num_parameters + num_states - num_latent_var],
                                  [self._batch_size,num_wave_plates])),
                                  axis=1)
                      
                  if delay+1 > threshold and threshold > 0:
                      for t in range(threshold-1):
                          angle = tf.concat((c_dict[t+1],
                                  tf.slice(inputs,[0, (2*delay+2-threshold + t) * \
                                           num_parameters + num_states - num_latent_var],
                                  [self._batch_size,num_wave_plates])),
                                  axis=1)
                          inp_c_past = tf.concat((inp_c_past,angle),axis=1)
                              
                  elif delay < threshold:
                      for t in range(threshold-delay-1,threshold-1):
                          if t == threshold-delay-1:
                              inp_c_past = tf.concat((c_dict[t+1],
                                  tf.slice(inputs,[0, (2*delay+2-threshold+t) * \
                                           num_parameters + num_states - num_latent_var],
                                          [self._batch_size,num_wave_plates])),
                                          axis=1)
                          else:
                              angle = tf.concat((c_dict[t+1],
                                      tf.slice(inputs,[0, (2*delay+2-threshold+t) * \
                                          num_parameters + num_states - num_latent_var],
                                          [self._batch_size,num_wave_plates])),
                                          axis=1)
                              inp_c_past = tf.concat((inp_c_past,angle),axis=1)
        
              if threshold < self._time_steps-1:
                  threshold += 1
              else:
                  threshold = 0
          
          # the network structure of the RNN for MPC tasks
          with tf.name_scope('hidden_layer_lp'):
              act_lp = tf.nn.relu(tf.matmul(inp_p_cur, W_lp_current) + \
                                      tf.matmul(inp_p_past, W_lp_past) + B_lp)
              act_lp_1 = tf.nn.relu(tf.matmul(act_lp, W_h[0]) + B_h[0])
          with tf.name_scope('hidden_layer_lc'):
              act_lc = tf.nn.relu(tf.matmul(inp_c_cur, W_lc_current) + \
                              tf.matmul(inp_c_past, W_lc_past) + B_lc)
              act_lc_1 = tf.nn.relu(tf.matmul(act_lc, W_h[1]) + B_h[1])
          with tf.name_scope('hidden_layer_c'):
              act_c = tf.nn.relu(tf.matmul(inp_c_cur, W_c_current) + \
                             tf.matmul(inp_c_past, W_c_past) + B_c)
              act_c_1 = tf.nn.relu(tf.matmul(act_c, W_h[2]) + B_h[2])
          with tf.name_scope('hidden_layer_f'):
              act_f = tf.nn.relu(tf.matmul(inp_f_cur, W_f_current) + \
                             tf.matmul(inp_f_past, W_f_past) + B_f)
              act_f_1 = tf.nn.relu(tf.matmul(act_f, W_h[3]) + B_h[3])
          with tf.name_scope('latent_layer_l'):
              """
              input_lstm = tf.reshape((tf.matmul(h, W_ll) + \
                        tf.matmul(tf.multiply(act_lp_1,act_lc_1),W_lo) + B_lo),
                        shape=[1,self._batch_size,latent_layer_size])
              with tf.variable_scope('LSTM_Cell') as scope:
                  if self._lstm_reuse:
                      scope.reuse_variables()
                  rnn_out, rnn_sta = tf.nn.dynamic_rnn(
                                            latent_cell, inputs=input_lstm,
                                            initial_state=initial_state_latent,
                                            time_major=True)
                  lstm_variables = [v for v in tf.global_variables()
                    if v.name.startswith(scope.name)]
              act_lo = tf.reshape(rnn_out,shape=[self._batch_size,
                                                 latent_layer_size])
              """
              act_lo = tf.nn.relu(tf.matmul(h, W_ll) + \
                       tf.matmul(tf.multiply(act_lp_1,act_lc_1),W_lo) + B_lo)
              if (self._time_steps == 1 and self._batch_size == 200):
                  obs_act_lo.append(act_lo)
          with tf.name_scope('hidden_layer_l'):
              act_l = tf.nn.relu(tf.matmul(act_lo, W_l) + B_l)
              if (self._time_steps == 1 and self._batch_size == 200):
                  obs_act_l.append(act_l)
              act_l_1 = tf.nn.relu(tf.matmul(act_l, W_h[4]) + B_h[4])
              if (self._time_steps == 1 and self._batch_size == 200):
                  obs_act_l1.append(act_l_1)
          with tf.name_scope('output_layer'):
              # the NN output will be splited into two parts. The first part
              # consists of the states of the laser which contains all 
              # information needed to calculate the objective function value. 
              # These states depends on the control input. The prediction of 
              # the latent variable, however, does not depend on the control
              # input and, thus, is predicted without the influence of the 
              # control input
              #"""
              dec_states = tf.matmul(tf.multiply(tf.multiply(act_f_1, act_c_1),
                                            act_l_1), W_o) + B_o
              dec_latent = tf.matmul(tf.multiply(act_c_1, act_l_1), W_ol) + B_ol
              
              # 
              #act_o = tf.matmul(tf.multiply(tf.multiply(act_f_1, act_c_1),
              #                              act_l_1), W_o) + B_o
              
              if self._control:
                  # during the control task we want to avoid that the system 
                  # divergate. Thus, we clip the decoder output by minimum and
                  # maximum values
                  
                  energy, momentum = tf.split(dec_states, num_or_size_splits=2, 
                                              axis=1)
                  
                  tf.clip_by_value(energy, ener_min, ener_max)
                  tf.clip_by_value(momentum, moment_min, moment_max)
                  tf.clip_by_value(dec_latent, k_min, k_max)
                  
                  act_o = tf.concat([energy,momentum,dec_latent],axis=1)
              else:
                  act_o = tf.concat([dec_states,dec_latent],axis=1)
              """
              energy, momentum = tf.split(dec_states, num_or_size_splits=2, 
                                              axis=1)
                  
              tf.clip_by_value(energy,one, ten)
              tf.clip_by_value(momentum,one, ten)
              tf.clip_by_value(dec_latent,one, ten)
                  
              act_o = tf.concat([energy,momentum,dec_latent],axis=1)
              """ 
              observation_dec.append(act_o)
              
          new_h = act_lo
      
          new_c = act_o
          
          new_state = (new_c, new_h)
          
          return new_c, new_state


#%%
###############################################################################
##                    Initialization of RNN Cell inputs                      ##
###############################################################################

def RNN_inputs(time_steps, num):
    with tf.name_scope('input_RNN'):
        encoder_inp['encoder_inp_' + str(num)] = [tf.placeholder(
                            shape=[None,num_inputs_RNN], 
                            dtype=tf.float32, 
                            name='enc_{}'.format(t))
                            for t in range(time_steps)]
    
        # labels that represent the real outputs
        x_true['x_true_' + str(num)] = [tf.placeholder(
                            shape=[None,num_states],dtype=tf.float32,
                            name='true_{}'.format(t))
                            for t in range(time_steps)]
    
        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        decoder_inp['decoder_inp_' + str(num)] = [tf.placeholder(
                    shape=[None,num_inputs_RNN], 
                    dtype=tf.float32, 
                    name='GO')]
        for t in range(1,time_steps):
            decoder_inp['decoder_inp_' + str(num)].append(tf.placeholder(
                    shape=[None,num_inputs_RNN], 
                    dtype=tf.float32, 
                    name='dec_{}'.format(t)))
        #decoder_inp['decoder_inp_' + str(num)] = [ tf.zeros_like(
        #      encoder_inp.get('encoder_inp_' + str(num))[0], dtype=tf.float32,
        #      name='GO') ] + encoder_inp.get('encoder_inp_' + str(num))[:-1]
        keep_prob['keep_prob_' + str(num)] = tf.placeholder(tf.float32)
        decoder_outputs['decoder_outputs_' + str(num)] = [tf.placeholder(
                       shape=[None,num_states], 
                       dtype=tf.float32, 
                       name='dec_out_{}'.format(t)) for t in range(time_steps)]


#%%
###############################################################################
##                       Initialization of RNN Cell                          ##
###############################################################################

output_list = []

def RNN_cell_initialization(time_steps,
                            num, control=False,
                            lstm_reuse=False,
                            batch_size=100):
    RNN_inputs(time_steps, num)
    with tf.name_scope('RNN_cell_initialization'):
            cell = myRNNCell(num_units=num_states, state_is_tuple=True,
                             control=control,lstm_reuse=lstm_reuse,
                             batch_size=batch_size, time_steps=time_steps)
            basic_cell['basic_cell_' + str(num)] = \
                tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                cell, output_keep_prob=keep_prob.get('keep_prob_' + str(num)))
                # tf.contrib.rnn.DropoutWrapper(
                
            # stack cells together : n layered model
            # stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(
            #       [basic_cell]*time_steps, state_is_tuple=True)
        
            with tf.name_scope('seq2seq'):
                
                # This model first runs an RNN to encode encoder_inputs into a 
                # state vector, and then runs decoder, initialized with the last 
                # encoder state, on decoder_inputs. Encoder and decoder use the 
                # same RNN cell and share parameters.
                decoder_outputs['decoder_outputs_' + str(num)], decoder_states[
                        'decoder_states_' + str(num)] = \
                tf.contrib.legacy_seq2seq.tied_rnn_seq2seq(
                    encoder_inputs=encoder_inp.get('encoder_inp_' + str(num)),
                    decoder_inputs=decoder_inp.get('decoder_inp_' + str(num)),
                    cell=basic_cell.get('basic_cell_' + str(num)),
                    dtype=tf.float32)
                    # .get('basic_cell_' + str(num))
                output_list.append(decoder_outputs['decoder_outputs_' + str(num)])

 
#%%
loss_list = []
added_loss = []


def train(name, num, time_steps,control=False,
          lstm_reuse=False,batch_size=100):
    # predict future states x
    """ Train Model Predictive (MP) Neural Network (NN).
    
    Parameters
    --------------------------------------------------------------------------
    decoder_outputs['decoder_outputs_{1,11,2,12,3}']:,
        tensorflow placeholder, shape = [None, num_wave_plates]
        output of the neural network
        
    x_true['x_true_{1,11,2,12,3}']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        true output
                
    loss['loss_{1,11,2,12,3}']:
        tensorflow placeholder, shape = [None, ]
        L2 norm loss function defined as: output = sum(t ** 2) / 2
    
    error['error_{1,11,2,12,3}']:
        tensorflow placeholder, shape = [None, ]
        average error above the training/test set defined as: mean(l2_norm(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5')))
            
    max_error['max_error_{1,11,2,12,3}']:
        tensorflow placeholder, shape = [None, ]
        maximum error defined as: max(sum(square(
            decoder_outputs['decoder_outputs_{1,11,2,12,3}'] - \
            x_true.get('x_true_{1,11,2,12,3}'))))
    
    Results
    --------------------------------------------------------------------------
    This function creates the training procedure for Model Predictive Neuran 
    Network. The MP architecture is used to predict the future states of the
    dynamical system. The loss function for the learning task is the L2 Norm
    while for the control task, the negative objective function will be 
    minimized. For optimizing the NN, the Adams Algorithm is used.

    """
    
    list_of_trainables = [W_lc_current, W_lc_past, B_lc, W_lp_current,
                          W_lp_past, B_lp, W_c_current, W_c_past, B_c,
                          W_f_current, W_f_past, B_f, W_l, B_l, W_lo, B_lo,
                          W_o, W_ol, B_o, B_ol, W_ll] #, lstm_variables[0],
                          #lstm_variables[1]
    for i in range(5):
        list_of_trainables.append(W_h[i])
        list_of_trainables.append(B_h[i])
    
    if not control:
        with tf.name_scope('loss_function'):
            loss['loss_' + str(num)] = 0.0
            max_error['max_error_' + str(num)] = 0.0
            for pred_val, true_val in zip(
                    decoder_outputs.get('decoder_outputs_'  + str(num)),
                    x_true.get('x_true_' + str(num))):
                
                loss['loss_' + str(num)] += tf.reduce_mean(tf.nn.l2_loss(
                                             tf.subtract(pred_val,true_val)))
                
                time_max_err = tf.reduce_max(tf.reduce_sum(tf.square(
                                             tf.subtract(pred_val,true_val)),
                                             axis=1))
                max_error['max_error_' + str(num)] += time_max_err

                
            #        tf.maximum(max_error.get('max_error_' + str(num)), temp)
            #loss_weights['loss_weights_' + str(num)] = [tf.ones_like(
            #   abel, 
            #  dtype=tf.float32) for label in x_true.get('x_true_' + str(num))]
            #loss['loss_' + str(num)] = sequence_loss(
            #               decoder_outputs.get('decoder_outputs_' + str(num)),
            #                   x_true.get('x_true_' + str(num)),
            #                   loss_weights.get('loss_weights_' + str(num)))
            #loss['loss_' + str(num)] = tf.nn.l2_loss(tf.subtract(
            #       decoder_outputs.get('decoder_outputs_' + str(num)),
            #       x_true.get('x_true_' + str(num))))
            tf.summary.scalar('loss_function', loss.get('loss_' + str(num)))
            tf.summary.scalar('max_error',max_error.get('max_error_' + str(num)))
        
        if num < 10:
            with tf.name_scope('train_phase'):
                with tf.variable_scope(name):
                    # Optimization algorithm:
                    # AdagradOptimizer(FLAGS.learning_rate, name='AdaGrad')
                    # RMSPropOptimizer(FLAGS.learning_rate,decay=0.9,
                    #                  momentum=FLAGS.momentum,
                    #                  name='RMSProp_control')
                    # ...
                    optimizer = tf.train.AdamOptimizer(learning_rate=0.01,
                                                       beta1=0.95, beta2=0.999,
                                                       epsilon=1e-3)
                    train_step['train_step_' + str(num)] = optimizer.minimize(
                                                    loss.get('loss_' + str(num)),
                                                    var_list=list_of_trainables)
            
    else:
        # Identification of the best angles
        """
        with tf.name_scope('loss_function'):
            for pred_val, true_val in zip(
            decoder_outputs.get('decoder_outputs_'  + str(num)),
            x_true.get('x_true_' + str(num))):
                pred_1,pred_2,_ = tf.split(pred_val, num_or_size_splits=3,
                                                   axis=1)
                #true_1,true_2,true_3 = tf.split(true_val, 
                #                                num_or_size_splits=3,
                #                                axis=1)
                
                loss_list.append(tf.reduce_mean(
                        -1.0*tf.nn.l2_loss(pred_1) + tf.nn.l2_loss(pred_2)))
                #added_loss.append(tf.minimum(
                #        tf.reduce_min(control_input[t])+np.pi, 0) - \
                #        tf.maximum(tf.reduce_max(control_input[t])-np.pi, 0))
            loss['loss_' + str(num)] = tf.reduce_min(
                    (tf.reduce_mean(loss_list), limit),0) #+ \
                    #0.1*tf.reduce_max(added_loss)
            tf.summary.scalar('Objective_function', loss.get('loss_' + str(num)))
        with tf.name_scope('train_control'):
            # Optimization algorithm:
            # AdagradOptimizer(FLAGS.learning_rate, name='AdaGrad')
            # RMSPropOptimizer(FLAGS.learning_rate,decay=0.9, 
            #                  momentum=FLAGS.momentum, name='RMSProp_control')
            # ...
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001,
                                               beta1=0.95,beta2=0.999,
                                               epsilon=1e-3)
            grads = optimizer.compute_gradients(loss.get('loss_' + str(num)),
                                                var_list=control_input)
            clipped_grads = [(tf.clip_by_value(grad, -2.0, 2.0), var) for grad,
                             var in grads]
            optimizer.apply_gradients(clipped_grads)    
            train_step['train_step_' + str(num)] = optimizer.minimize(
                    loss.get('loss_' + str(num)), var_list=control_input)
        """
        with tf.name_scope('loss_function'):
            for t in range(time_steps):
                split_1,split_2,_ = tf.split(
                        decoder_outputs.get('decoder_outputs_' + str(num))[t],
                        num_or_size_splits=3, axis=1)
                #split_1 = tf.add(tf.scalar_mul(data_std[0], split_1),
                #                 mean_Energy)
                #split_2 = tf.add(tf.scalar_mul(data_std[1], split_2),
                #                 mean_Kurtosis)
                loss_list.append(tf.reduce_mean(
                        tf.div(split_1,tf.maximum(split_2,minK))))
                #added_loss.append(tf.minimum(
                #        tf.reduce_min(control_input[t])+np.pi, 0) - \
                #        tf.maximum(tf.reduce_max(control_input[t])-np.pi, 0))
            loss['loss_' + str(num)] = tf.reduce_min(
                    (tf.reduce_mean(loss_list), limit),0) #+ \
                    #0.1*tf.reduce_max(added_loss)
            tf.summary.scalar('Objective_function', loss.get('loss_' + str(num)))
        with tf.name_scope('train_control'):
            # Optimization algorithm:
            # AdagradOptimizer(FLAGS.learning_rate, name='AdaGrad')
            # RMSPropOptimizer(FLAGS.learning_rate,decay=0.9, 
            #                  momentum=FLAGS.momentum, name='RMSProp_control')
            # ...
            
            """ ### CHECK IF CLIPPING THE GRADS IS NECESSARY -> ALREADY CLIPPED
            THE DECODER OUTPUT BY A VALUE ### """
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
                                               beta1=0.95,beta2=0.999,
                                               epsilon=1e-3)
            grads = optimizer.compute_gradients(loss.get('loss_' + str(num)),
                                                var_list=control_input)
            clipped_grads = [(tf.clip_by_value(grad, -2.0, 2.0), var) for grad,
                             var in grads]
            optimizer.apply_gradients(clipped_grads)    
            train_step['train_step_' + str(num)] = optimizer.minimize(
                    -1.0*loss.get('loss_' + str(num)), var_list=control_input)
        
        # Updating the weights
        with tf.name_scope('Update'):
            with tf.name_scope('loss_function'):
                loss['loss_' + str(num+1)] = 0
                for pred_val, true_val in zip(
                        decoder_outputs.get('decoder_outputs_'  + str(num)),
                        x_true.get('x_true_' + str(num))):
                    loss['loss_' + str(num+1)] += tf.reduce_mean(tf.nn.l2_loss(
                                            tf.subtract(pred_val,true_val)))
                #loss_weights['loss_weights_' + str(num+1)] = [tf.ones_like(
                #    label, 
                #    dtype=tf.float32) for label in x_true.get('x_true_' + str(num))]
                #loss['loss_' + str(num+1)] = sequence_loss(
                #                decoder_outputs.get('decoder_outputs_' + str(num)),
                #                    x_true.get('x_true_' + str(num)),
                #                    loss_weights.get('loss_weights_' + str(num+1)))
                tf.summary.scalar('loss_function',
                                  loss.get('loss_' + str(num+1)))
            with tf.name_scope('train_phase'):
                with tf.variable_scope(name):
                    # Optimization algorithm:
                    # AdagradOptimizer(FLAGS.learning_rate, name='AdaGrad')
                    # RMSPropOptimizer(FLAGS.learning_rate,decay=0.9, 
                    #                  momentum=FLAGS.momentum,
                    #                  name='RMSProp_control')
                    # ...
                    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001,
                                                       beta1=0.95,
                                                       beta2=0.999,
                                                       epsilon=1e-3)
                    train_step['train_step_' + str(num+1)] = \
                    optimizer.minimize(loss.get('loss_' + str(num+1)),
                                       var_list=list_of_trainables)
            
            with tf.name_scope('error'):
                error['error_' + str(num+1)] = \
                loss.get('loss_' + str(num+1))/batch_list.get(
                                               'batch_' + str(num))
            tf.summary.scalar('error', error.get('error_' + str(num+1)))
            
            # Merge all the summaries and write them out to
            merged['merged_' + str(num+1)] = tf.summary.merge(
                [tf.summary.scalar('error', error.get('error_' + str(num+1))),
                 tf.summary.scalar('loss_function',
                                   loss.get('loss_' + str(num+1)))])
                #merge_all
        
        
    with tf.name_scope('error'):
        error['error_' + str(num)] = \
        loss.get('loss_' + str(num))/batch_list.get('batch_' + str(num))
    tf.summary.scalar('error', error.get('error_' + str(num)))
    
    # Merge all the summaries and write them out to
    if not control:
        merged['merged_' + str(num)] = tf.summary.merge(
            [tf.summary.scalar('error', error.get('error_' + str(num))),
             tf.summary.scalar('max_error',max_error.get('max_error_' + str(num))),
             tf.summary.scalar('loss_function', loss.get('loss_' + str(num)))])
    else:
        merged['merged_' + str(num)] = tf.summary.merge(
            [tf.summary.scalar('error', error.get('error_' + str(num))),
             tf.summary.scalar('loss_function', loss.get('loss_' + str(num)))])
       
 
#%%
###############################################################################
##                            Build RNN Graph                                ##
###############################################################################

training_phases = ['one_step_pred', 'ten_steps_pred', 'test_pred_1',
                   'test_pred_2', 'control_pred']
num_phase = [1, 2, 11, 12, 3]
batch_sizes = [train_batch_size, train_batch_size, test_batch_1, 
               test_batch_2, control_batch_size]
steps_per_phase= [steps_phase_1, steps_phase_2,  steps_phase_1, steps_phase_2, 
                  steps_phase_crtl]
crtl_list = [False, False, False, False, True]
# After created lstm trainables, the scope has to be set to reuse
lstm_reuse = [False, True, True, True, True]
    
with tf.name_scope('Variables'):
    with tf.name_scope('Denormalization'):
        with tf.name_scope('Energy'):
            mean_Energy = tf.Variable(initial_value=data_mean[0],
                                      dtype=tf.float32,
                                      trainable=False)
    
        with tf.name_scope('Kurtosis'):
            mean_Kurtosis = tf.Variable(initial_value=data_mean[1],
                                        dtype=tf.float32,
                                        trainable=False)
        with tf.name_scope('Birefringence'):
            mean_K = tf.Variable(initial_value=0,
                                 dtype=tf.float32,
                                 trainable=False)
    
    with tf.name_scope('Constraint'):
        with tf.name_scope('dec_output_limits'):
            ener_min = tf.Variable(initial_value=data_min[0], trainable=False, 
                                   dtype=tf.float32)
            moment_min = tf.Variable(initial_value=data_min[1], 
                                     trainable=False, dtype=tf.float32)
            k_min = tf.Variable(initial_value=data_min[2], trainable=False, 
                                dtype=tf.float32)
            ener_max = tf.Variable(initial_value=data_max[0], trainable=False, 
                                   dtype=tf.float32)
            moment_max = tf.Variable(initial_value=data_max[1], 
                                     trainable=False, dtype=tf.float32)
            k_max = tf.Variable(initial_value=data_max[2], trainable=False, 
                                dtype=tf.float32)
            one = tf.Variable(initial_value=1, trainable=False, 
                                dtype=tf.float32)
            nine = tf.Variable(initial_value=9, trainable=False, 
                                dtype=tf.float32)
            ten = tf.Variable(initial_value=10, trainable=False, 
                                dtype=tf.float32)
        with tf.name_scope('placeholders'):  
            c_dict = [tf.get_variable(name='vars_{}'.format(t),
                              dtype=tf.float32,trainable=False,
                              initializer=tf.random_uniform(
                              shape=[control_batch_size,num_states - num_latent_var],
                              dtype=tf.float32))
                      for t in range(steps_phase_crtl)]

with tf.Session() as sess:
    for name,num,steps,crtl,lstm_r,batch in zip(training_phases,num_phase,
                                                steps_per_phase, crtl_list,
                                                lstm_reuse, batch_sizes):
        with tf.name_scope(name):
            if name == 'control_pred':
                with tf.name_scope('One'):
                    minK = tf.get_variable(name='min_kurtosis',shape=[1,1],
                                       initializer=tf.constant_initializer(0.1),
                                       dtype=tf.float32,
                                       trainable=False)
                    limit = tf.Variable(initial_value=100, dtype=tf.float32)
            RNN_cell_initialization(steps, num, control=crtl,
                                    lstm_reuse=lstm_r, batch_size=batch)
            train(name,num,time_steps=steps, control=crtl,
                      lstm_reuse=lstm_r, batch_size=batch)


#%%
"""
###############################################################################
##                            Feed-Dict generation                           ##
###############################################################################
"""

iter_data_2 = 0
iter_through_dataset_2 = 1

def feed_inp(phase, time_steps,train_batch_size,train, num_batch = 0):
    """ Generating the feed_dict from the data set
    
    Parameters
    --------------------------------------------------------------------------
    phase:  int
            needed to refer to the respective tensorflow placeholders
        
    time_steps:
            int
            depending on how many time steps into the future the prediction 
            will be, the feed_dict will be adapted
                
    train_batch_size:
            int
            defines how large the batch will be
    
    num_batch:
            int
            due to memory limitations, not the whole test set can be used as
            feed_dict at the same time. It is splited into several parts which
            can be distinguished using 'num_batch'
            
    Results
    --------------------------------------------------------------------------
    This function returns a feed_dict which is used to train the model.

    """
    #    Division of the inputs:
    #    inp_p_past[t-11, t-10, t-9, t-8, t-7]
    #    inp_p_cur[t-6]
    #    inp_c_past[t-5, t-4, t-3, t-2, t-1]
    #    inp_c_cur[t]
    #    inp_f_past[t-4, t-3, t-2, t-1, t]
    #    inp_f_cur[t+1]
    
    # valid starting indices for training
    batchdataindex = []
    last = 0
    for s in [seqlen_train]:
            batchdataindex += range(last + 2*delay + 3,
                                    last + s-train_batch_size-2*time_steps-1)
            last += s
    
    permindex = np.array(batchdataindex)
    np.random.shuffle(permindex)
    
        
    def next_batch(phase, train, time_steps, train_batch_size):
            
            global iter_data_2
            global iter_through_dataset_2
            if iter_data_2 == iter_through_dataset_2*(
                    np.size(batchdataindex)-train_batch_size - 1): 
                iter_data_2 = 0
                #iter_through_dataset_2 += 1
                print(iter_data_2)
                print(iter_through_dataset_2)
                np.random.shuffle(permindex)
            batch_v_p_hist = np.zeros((2*time_steps,train_batch_size,
                                       delay*num_parameters))
            batch_v_p_cur = np.zeros((2*time_steps,train_batch_size,
                                      num_parameters))
            batch_v_hist = np.zeros((2*time_steps,train_batch_size,
                                     delay*num_parameters))
            batch_v_cur = np.zeros((2*time_steps,train_batch_size,
                                    num_parameters))
            batch_u_hist = np.zeros((2*time_steps,train_batch_size,
                                     delay*num_wave_plates))
            batch_u_cur = np.zeros((2*time_steps,train_batch_size,
                                    num_wave_plates))
            batch_u_comp_h = np.zeros((2*time_steps,train_batch_size,
                                       delay*num_parameters))
            batch_u_comp_c = np.zeros((2*time_steps,train_batch_size,
                                      num_parameters))
            batch_true = np.zeros((2*time_steps,train_batch_size,num_states))
            for num in range(train_batch_size):
                for t in range(2*time_steps):
                    batch_v_p_hist[t,num,:] = train_data[permindex[
                            iter_data_2 + num] - 2*delay-2 + t:permindex[
                            iter_data_2 + num]-delay-2 + t,
                            0:num_parameters].reshape([1,num_parameters*delay])
                    batch_v_p_cur[t,num,:] = train_data[permindex[
                            iter_data_2 + num]-delay-2 + t,
                            0:num_parameters].reshape([1,num_parameters])
                    batch_v_hist[t,num,:] = train_data[permindex[
                            iter_data_2 + num] - delay-1 + t:permindex[
                            iter_data_2 + num]-1 + t,0:num_parameters].reshape(
                            [1,num_parameters*delay])
                    batch_v_cur[t,num,:] = train_data[
                            permindex[iter_data_2 + num]-1 + t,
                            0:num_parameters].reshape([1,num_parameters])
                    batch_u_hist[t,num,:] = train_data[
                            permindex[iter_data_2 + num]-delay + \
                            t:permindex[iter_data_2 + num] + t,
                            num_states - num_latent_var:num_parameters].reshape(
                            [1,num_wave_plates*delay])
                    batch_u_cur[t,num,:] = train_data[
                            permindex[iter_data_2 + num] + t,
                            num_states - num_latent_var:num_parameters].reshape(
                            [1,num_wave_plates])
                    batch_u_comp_h[t,num,:] = train_data[
                            permindex[iter_data_2 + num] - delay + t:permindex[
                            iter_data_2 + num] + t,0:num_parameters].reshape(
                            [1,num_parameters*delay])
                    batch_u_comp_c[t,num,:] = train_data[
                            permindex[iter_data_2 + num] + t,
                            0:num_parameters].reshape([1,num_parameters])
                    batch_true[t,num,:] = train_data_K[
                            permindex[iter_data_2 + num] + t,
                            0:num_states].reshape([1,num_states])
                            #* data_std[:num_states]+data_mean[:num_states]
            
            iter_data_2 += 1
            
            return (batch_v_cur, batch_v_hist, batch_v_p_cur, batch_v_p_hist,
                    batch_u_cur, batch_u_hist, batch_u_comp_c, batch_u_comp_h,
                    batch_true)
    
        
    def feed_dictionary(phase,train,time_steps,train_batch_size,num_batch=0):
            if train:
                (batch_v_cur, batch_v_hist, batch_v_p_cur, batch_v_p_hist,
                 batch_u_cur, batch_u_hist, batch_u_comp_c, batch_u_comp_h,
                 batch_true) = next_batch(phase, train,time_steps,
                                          train_batch_size)
            else:
                # valid starting indices for testing
                testdataindex = range(2*delay+2,seqlen_test-2*time_steps+1)
                
                testindex = np.array(testdataindex)
                np.random.shuffle(testindex)
                
                batch_v_cur = np.zeros((2*time_steps,train_batch_size,
                                        num_parameters))
                batch_v_hist = np.zeros((2*time_steps,train_batch_size,
                                         delay*num_parameters))
                batch_v_p_cur = np.zeros((2*time_steps,train_batch_size,
                                          num_parameters))
                batch_v_p_hist = np.zeros((2*time_steps,train_batch_size,
                                           delay*num_parameters))
                batch_u_cur = np.zeros((2*time_steps,train_batch_size,
                                        num_wave_plates))
                batch_u_hist = np.zeros((2*time_steps,train_batch_size,
                                         delay*num_wave_plates))
                batch_u_comp_c = np.zeros((2*time_steps,train_batch_size,
                                      num_parameters))
                batch_u_comp_h = np.zeros((2*time_steps,train_batch_size,
                                           delay*num_parameters))
                batch_true = np.zeros((2*time_steps,train_batch_size,
                                       num_states))
                for num in range(train_batch_size):
                    for t in range(2*time_steps):
                        batch_v_p_hist[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num +  t:num_batch * \
                                      train_batch_size + num + delay + t,
                                      0:num_parameters].reshape(
                                      [1,num_parameters*delay])
                        batch_v_p_cur[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num + delay + t,
                                      0:num_parameters].reshape(
                                      [1,num_parameters])
                        batch_v_hist[t,num,:] = test_data[num_batch * \
                                     train_batch_size + num + delay + 1 + \
                                     t:num_batch * train_batch_size + num + \
                                     2*delay + 1 + t,
                                     0:num_parameters].reshape(
                                     [1,num_parameters*delay])
                        batch_v_cur[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num + 2*delay + 1 + t,
                                      0:num_parameters].reshape(
                                      [1,num_parameters])
                        batch_u_hist[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num + delay + 2 + \
                                      t:num_batch * train_batch_size + num + \
                                      2*delay + 2 + t,
                                      num_states - num_latent_var:num_parameters].reshape(
                                      [1,num_wave_plates*delay])
                        batch_u_cur[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num + 2*delay + 2 + t,
                                      num_states - num_latent_var:num_parameters].reshape(
                                      [1,num_wave_plates])
                        batch_u_comp_h[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num + delay + 2 + \
                                      t:num_batch * train_batch_size + num + \
                                      2*delay + 2 + t,
                                      0:num_parameters].reshape(
                                      [1,num_parameters*delay])
                        batch_u_comp_c[t,num,:] = test_data[num_batch * \
                                      train_batch_size + num + 2*delay + 3 + t,
                                       0:num_parameters].reshape(
                                       [1,num_parameters])
                        # since the latent values from the variational
                        # autoencoder results in small values with a even
                        # smaller variance, we use the normalized values for
                        batch_true[t,num,:] = test_data_K[num_batch * \
                                   train_batch_size + num + 2*delay + 3 + t,
                                   0:num_states].reshape([1,num_states])
                                    #* data_std[:num_states]+data_mean[:num_states]
            
            inp = np.concatenate((batch_v_p_hist, batch_v_p_cur, batch_v_hist,
                                  batch_v_cur,  batch_u_hist, batch_u_cur),
                                axis=2)
            feed_dict = {encoder_inp.get(
                    'encoder_inp_' + str(phase))[t]: inp[t,:,:] for t in range(
                            time_steps)}
            feed_dict.update({decoder_inp.get(
                      'decoder_inp_' + str(phase))[t]: inp[t+time_steps,:,:] 
                      for t in range(time_steps)})
            feed_dict.update({keep_prob.get('keep_prob_' + str(phase)): 1})
            
            feed_dict.update({x_true.get(
                      'x_true_' + str(phase))[t]: batch_true[t+time_steps,:,:]
                      for t in range(time_steps)})
            next_iter = (batch_v_p_hist[time_steps:,:,:],
                         batch_v_p_cur[time_steps:,:,:],
                         batch_v_hist, batch_v_cur,
                         batch_u_comp_h, batch_u_comp_c,
                         batch_u_hist, batch_u_cur)
            return feed_dict, next_iter, inp
     
    feed_dict, next_iter, inp = feed_dictionary(phase,
                                                train,
                                                time_steps,
                                                train_batch_size,
                                                num_batch)

    return feed_dict, next_iter, inp
    

#%%    
"""
###############################################################################
##                           One-Step Prediction                             ##
###############################################################################
"""

def one_step():
    """ Training of the One-Step Prediction.
    
    Parameters
    --------------------------------------------------------------------------
    decoder_outputs['decoder_outputs_1']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        output of the neural network
        
    x_true['x_true_1']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        true output
                
    loss['loss_1']:
        tensorflow placeholder, shape = [None, ]
        L2 norm loss function defined as: output = sum(t ** 2) / 2
    
    error['error_1']:
        tensorflow placeholder, shape = [None, ]
        average error above the training/test set defined as: mean(l2_norm(
            decoder_outputs['decoder_outputs_1'] - x_true.get('x_true_1')))
            
    max_error['max_error_1']:
        tensorflow placeholder, shape = [None, ]
        maximum error defined as: max(sum(square(
            decoder_outputs['decoder_outputs_1'] - x_true.get('x_true_1'))))
    
    Results
    --------------------------------------------------------------------------
    This function trains the one step prediction. The data set is 
    splitted into a training and a test set. Weights and biases are saved once 
    average error of the test set as well as the maximum error decreased.

    """
    i = 0
    err_testset = 100000
    max_err = 100000
    err_temp = err_testset
    max_err_temp = max_err
    total_batch = int(seqlen_test / test_batch_1)
    #for i in range(FLAGS.max_steps):
    while(i < FLAGS.max_steps_1 and err_testset > 5e-5 and max_err > 1e-3):
        if i % 200 == 0:  # Record summaries and test-set accuracy
            err_testset = 0
            max_err = 0
            for num in range(total_batch):
                feed_dict,_,_ = feed_inp(11, steps_phase_1, test_batch_1,
                                           False, num_batch=num)
                if num == 0:
                    summary, err_test, max_err_test = \
                        sess.run([merged.get('merged_11'),
                                  error.get('error_11'),
                                  max_error.get('max_error_11')],
                                  feed_dict=feed_dict)
                else:
                    err_test, max_err_test = sess.run([error.get('error_11'),
                                                max_error.get('max_error_11')],
                                                feed_dict=feed_dict)
                err_testset += err_test / total_batch
                max_err = np.max((max_err, max_err_test))
                
            test_writer.add_summary(summary, i)
            out, true = sess.run([output_list[2], x_true.get('x_true_11')],
                                  feed_dict=feed_dict)
            out = np.vstack(out)
            true = np.vstack(true)
            if i == 0:
                with open('pred_vs_true.csv', 'w',newline='') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerows(np.concatenate((out,true),axis=1))
                    spamwriter.writerow([err_testset])
            else:
                with open('pred_vs_true.csv', 'a',newline='') as csvfile:
                    spamwriter = csv.writer(csvfile)
                    spamwriter.writerows(np.concatenate((out,true),axis=1))
                    spamwriter.writerow([err_testset])
            print('Error at step %s: avg.:%s, max.: %s' % (i, err_testset, 
                                                           max_err))
            
            
            a = np.array((err_temp,max_err_temp))
            b = np.array((err_testset,max_err))
            rel = sum((a-b)/np.array((max(a[0],b[0]),max(a[1],b[1]))))
            if rel > 0:
                #err_testset < err_temp and max_err < max_err_temp
                print('+++ Save weights and biases +++')
                err_temp = err_testset
                max_err_temp = max_err
                if i == 0:
                    saver.save(sess, os.path.join(os.getcwd(),
                               'Save/trained_vars'), global_step=i)
                else:
                    saver.save(sess, os.path.join(os.getcwd(),
                               'Save/trained_vars'),write_meta_graph=False,
                               global_step=i)
            
        else:  # Record train set summaries, and train
            if i % 2000 == 9:  # Record execution stats
                run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                feed_dict,_,_ = feed_inp(1, steps_phase_1,
                                        train_batch_size, True)
                summary, _ = sess.run([merged.get('merged_1'),
                                       train_step.get('train_step_1')],
                              feed_dict=feed_dict,
                              options=run_options,
                              run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                #print('Adding run metadata for', i)
            else:  # Record a summary
                feed_dict,_,_ = feed_inp(1, steps_phase_1,
                                        train_batch_size, True)
                sess.run([merged.get('merged_1'),
                          train_step.get('train_step_1')],
                          feed_dict=feed_dict)
                #train_writer.add_summary(summary, i)
                
        i += 1
                
    train_writer.close()
    test_writer.close()
    
    parameter_list = [v for v in tf.trainable_variables()]
    model_parameter = tuple(parameter_list[0:(np.size(parameter_list)-1)//2])
    
    return model_parameter

#%%
###############################################################################
##                            Tensorboard writer                             ##
###############################################################################

saver = tf.train.Saver()
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

if not reuse_model:
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        sess.run(angles_min.assign(data_min[num_states:]))
        sess.run(angles_max.assign(data_max[num_states:]))
        
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        train_writer_2 = tf.summary.FileWriter(FLAGS.log_dir + '/train_2')
        test_writer_2 = tf.summary.FileWriter(FLAGS.log_dir + '/test_2')
        train_writer_cnt = tf.summary.FileWriter(FLAGS.log_dir + '/train_cnt')
        train_writer_cnt_2 =tf.summary.FileWriter(FLAGS.log_dir+'/train_cnt_2')
        train_writer_3 = tf.summary.FileWriter(FLAGS.log_dir + '/train_3')
        test_writer_3 = tf.summary.FileWriter(FLAGS.log_dir + '/test_3')
        
        model_parameter = one_step()
#%%
"""
###############################################################################
##                           Ten-Steps Prediction                            ##
###############################################################################
"""

def ten_steps():
    """ Training of the Ten-Steps Prediction.
    
    Parameters
    --------------------------------------------------------------------------
    decoder_outputs['decoder_outputs_2']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        output of the neural network
        
    x_true['x_true_2']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        true output
                
    loss['loss_2']:
        tensorflow placeholder, shape = [None, ]
        L2 norm loss function defined as: output = sum(t ** 2) / 2
    
    error['error_2']:
        tensorflow placeholder, shape = [None, ]
        average error above the training/test set defined as: mean(l2_norm(
            decoder_outputs['decoder_outputs_2'] - x_true.get('x_true_2')))
            
    max_error['max_error_2']:
        tensorflow placeholder, shape = [None, ]
        maximum error defined as: max(sum(square(
            decoder_outputs['decoder_outputs_2'] - x_true.get('x_true_2'))))
    
    Results
    --------------------------------------------------------------------------
    This function trains the ten steps prediction. The data set is 
    splitted into a training and a test set. Weights and biases are saved once 
    average error of the test set as well as the maximum error decreased.

    """
    print('Start ten steps prediction...')
    epoch = 0
    err_testset = 100000
    max_err = 100000
    err_temp = err_testset
    max_err_temp = max_err
    total_batch = int(seqlen_test / test_batch_1) - 1
    #for epoch in range(FLAGS.max_steps):
    while(epoch < FLAGS.max_steps_10 and err_testset > 1e-3 and max_err > 1e-1):
            if (epoch) % 200 == 0:  # Record summaries and test-set accuracy
                err_testset = 0
                max_err = 0
                for num in range(total_batch):
                    feed_dict,_,_ = feed_inp(12, steps_phase_2,test_batch_2,
                                               False, num_batch=num)
                    if num == 0:
                        summary, err_test, max_err_test = \
                            sess.run([merged.get('merged_12'),
                                      error.get('error_12'),
                                      max_error.get('max_error_12')],
                                      feed_dict=feed_dict)
                    else:
                        err_test,max_err_test = sess.run([error.get('error_12'),
                                                max_error.get('max_error_12')],
                                                feed_dict=feed_dict)
                    err_testset += err_test / total_batch
                    max_err = np.max((max_err, max_err_test))
                    
                test_writer_2.add_summary(summary, epoch)
                print('Error in iteration %s: avg.: %s, max.: %s' % (epoch,
                                                               err_testset,
                                                               max_err))
                a = np.array((err_temp,max_err_temp))
                b = np.array((err_testset,max_err))
                rel = sum((a-b)/np.array((max(a[0],b[0]),max(a[1],b[1]))))
                if rel > 0:
                    #err_testset < err_temp and max_err < max_err_temp
                    print('+++ Save weights and biases +++')
                    err_temp = err_testset
                    max_err_temp = max_err
                    saver.save(sess, os.path.join(os.getcwd(),
                                                  'Save/trained_vars'),
                                write_meta_graph=False, global_step=epoch)
                                
            else:
                if (epoch) % 2000 == 9:  # Record execution stats
                    run_options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                    feed_dict,_,_ = feed_inp(2,steps_phase_2,
                                             train_batch_size, True)
                    summary,_ = sess.run([merged.get('merged_2'),
                                          train_step.get('train_step_2')],
                                          feed_dict=feed_dict,
                                          options=run_options)
                    train_writer_2.add_summary(summary, epoch)
                else:  # Record a summary
                    feed_dict,_,_ = feed_inp(2,steps_phase_2,
                                             train_batch_size, True)
                    sess.run([merged.get('merged_2'),
                              train_step.get('train_step_2')],
                             feed_dict=feed_dict)
                    
            epoch += 1
                    
    train_writer_2.close()
    test_writer_2.close()
    
    parameter_list_2 = [v for v in tf.trainable_variables()]
    model_parameter_2 = tuple(parameter_list_2[0:(np.size(
                                                  parameter_list_2)-1)//2])
    
    return model_parameter_2

#%%
"""
###############################################################################
##                             Reuse best Model                              ##
###############################################################################
""" 
if not reuse_model:
    with tf.Session(config=config) as sess:
     
        sess.run(tf.global_variables_initializer())
        
        print(sess.run(W_lc_current))
        
        saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),
                                                                   './Save/')))
        
        trainables = tf.trainable_variables()
        print(sess.run(trainables[0]))
        print(sess.run(W_lc_current))
           
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
        train_writer_2 = tf.summary.FileWriter(FLAGS.log_dir + '/train_2')
        test_writer_2 = tf.summary.FileWriter(FLAGS.log_dir + '/test_2')
        train_writer_cnt = tf.summary.FileWriter(FLAGS.log_dir + '/train_cnt')
        train_writer_cnt_2 =tf.summary.FileWriter(FLAGS.log_dir+'/train_cnt_2')
        train_writer_3 = tf.summary.FileWriter(FLAGS.log_dir + '/train_3')
        test_writer_3 = tf.summary.FileWriter(FLAGS.log_dir + '/test_3')
        
        model_parameter_2 = ten_steps()
 

#%%
"""
###############################################################################
##                            Control Prediction                             ##
###############################################################################
"""
track = []
track_map_out = []
Ks_comp = []

def control_pred(Ks, pre_learning=True):
    """ Control Prediction.
    
    Parameters
    --------------------------------------------------------------------------
    decoder_outputs['decoder_outputs_2']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        output of the neural network
        
    x_true['x_true_2']:
        tensorflow placeholder, shape = [None, num_wave_plates]
        true output
                
    loss['loss_2']:
        tensorflow placeholder, shape = [None, ]
        L2 norm loss function defined as: output = sum(t ** 2) / 2
    
    error['error_2']:
        tensorflow placeholder, shape = [None, ]
        average error above the training/test set defined as: mean(l2_norm(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5')))
            
    max_error['max_error_2']:
        tensorflow placeholder, shape = [None, ]
        maximum error defined as: max(sum(square(
            decoder_outputs['decoder_outputs_5'] - x_true.get('x_true_5'))))
    
    Results
    --------------------------------------------------------------------------
    This function trains the one step prediction. The data set is 
    splitted into a training and a test set. Weights and biases are saved once 
    average error of the test set as well as the maximum error decreased.

    """
    # initial conditions for the laser simulation - determine uvt
    T = 60
    n = 256
    t2 = np.linspace(-T/2,T/2,n+1)
    t_dis = t2[0:n].reshape([1,n])      # time discretization
    u=np.reshape(sech(t_dis/2), [n,])   # orthogonally polarized electric field 
    v=np.reshape(sech(t_dis/2), [n,])   # envelopes in the optical fiber
    ut=np.fft.fft(u).reshape(n,)        # fast fourier transformation of the
    vt=np.fft.fft(v).reshape(n,)        # electrical fields
    uvt=np.concatenate([ut, vt], axis=0)# concatenation of ut and vt
    
    # If it is not the pre learning for the controlling, the initial feed_dict
    # has to be determined. Here for the first (2*delay + 2 + stepsphase_crtl) 
    # time steps, the states of the laser have to be calculated using the 
    # laser simulation
    if not pre_learning:
        #data_mean_ctrl = np.concatenate((data_mean[:num_states-1],[K_steps_mean],
        #                                 data_mean[num_states:]))
        #data_std_ctrl = np.concatenate((data_std[:num_states-1],[K_steps_std],
        #                                        data_std[num_states:]))
        res_past = 0.0
        alpha_past = 0
        
        init_states = np.zeros([2*steps_phase_crtl,1,
                     (2*delay + 2 + steps_phase_crtl)*num_parameters])
        init_angles = np.zeros([2*steps_phase_crtl,1,
                     (delay + 1 + steps_phase_crtl)*num_wave_plates])
        for t in range(2*delay + 2 + steps_phase_crtl):
            print('K-value: %s' % Ks[t])
            alpha_norm = sess.run([layer[-1]],
                               feed_dict={K: np.reshape(Ks[t],[1,1])})[0]
            
            alpha = alpha_norm*K_u_std[1:5]+K_u_mean[1:5]
            
            
            (_,states) = laser_simulation(uvt, alpha[0,0], alpha[0,1], 
                                          alpha[0,2], alpha[0,3], K_sim[t])
            
            print('Laser states: %s' % states)
            z_vae = vae.transform(np.reshape(states,[1,num_parameters]))
            K_vae = (z_vae - z_mu_mean)/z_mu_std
            #K_vae = vae.transform(np.reshape(states,[1,num_parameters-1]))k_VAE
            print('K-values: %s vs %s' % (K_vae[:,k_VAE_index], Ks[t]))
            # normalize states
            states = (states - data_mean_ctrl) / data_std_ctrl
            #states = np.concatenate((states[:num_states-1],
            #                         K_vae[:,k_VAE_index],
            #                         states[num_states-1:]),axis=0)
            
            init_states[0,0,t*num_parameters:(t+1)*num_parameters] = states
            if t > delay + 1:
                init_angles[0,0,(t-delay-2)*num_wave_plates:(t-delay-1)*num_wave_plates] = \
                states[num_states - num_latent_var:]
                
        for t in range(1,2*steps_phase_crtl):
            init_states[t,0,0:(2*delay + 2 + steps_phase_crtl - t)*num_parameters] = \
            init_states[0,0,num_parameters*t:(2*delay + 2 + steps_phase_crtl)*num_parameters]
            if t < (delay + 1 + steps_phase_crtl):
                init_angles[t,0,0:(delay + 1 + steps_phase_crtl - t)*num_wave_plates] = \
                init_angles[0,0,t*num_wave_plates:(delay + 1 + steps_phase_crtl)*num_wave_plates]
         
        inp_ctrl = np.concatenate((init_states[:,:,:((2*delay + 2)*num_parameters)],
                                   init_angles[:,:,:((delay + 1)*num_wave_plates)]),
                                   axis=2)
        track.append(inp_ctrl)
                
        feed_dict = {}
        feed_dict = {encoder_inp.get(
            'encoder_inp_3')[t]: inp_ctrl[t,:,:] for t in range(steps_phase_crtl)}
        feed_dict.update({decoder_inp.get(
            'decoder_inp_3')[t]: inp_ctrl[t+steps_phase_crtl,:,:] for t in range(
                    steps_phase_crtl)})
        feed_dict.update({keep_prob.get('keep_prob_3'): 1})
        feed_dict.update({x_true.get('x_true_3')[t]: np.array([[0,0,0]])
                                     for t in range(steps_phase_crtl)})
        
        time_span = 2*delay + 2 + steps_phase_crtl
        if len(Ks) < time_span:
            print('There have to be at least %s K values' % time_span)
            sys.exit()
        else:
            length = (len(Ks)-time_span)//steps_phase_crtl
        Kvals = np.zeros([length, steps_phase_crtl])
        Kvals_sim = np.zeros([length, steps_phase_crtl])
        for i in range(length):
            Kvals[i,:] = Ks[time_span + i * steps_phase_crtl:
                            time_span + (i + 1) * steps_phase_crtl]
            Kvals_sim[i,:] = K_sim[time_span + i * steps_phase_crtl:
                            time_span + (i + 1) * steps_phase_crtl]
                
    # for loop to go through the temporarly changing birefringence values
    for k, K_simulation, num in zip(Kvals, Kvals_sim, range(length)):
        # if it is not the pre learning step, at first the birefringence value 
        # K will be predicted by using the RNN cell and the before identified 
        # feed_dict. And the K value will be again the input for the 
        # K_u_mapping NN to determine the initial control inputs. 
        #
        # In case steps_phase_crtl > 1, this has to be done iteratively:
        #    for t in range(steps_phase_crtl):
        #        1. Predict K[t] using RNN Cell
        #        2. Predict u[t] using K_u_mappping and K[t] as input
        #
        # birefringence value k for comparison with the predicted one
        ks = k
        max_steps = 1
        if num == 0:
            alpha_inp = (alpha-data_mean[num_states:])/data_std[num_states:]
            #map_out = (map_out-K_u_mean[1:5])/K_u_std[1:5]
        
        track_map_out.append(alpha_inp)
        print('Alpha_inp: %s' % alpha_inp)
        sess.run(control_input[0].assign(np.reshape(alpha_inp,
                 [1,num_wave_plates])))
        for t in range(1,steps_phase_crtl):
            out = sess.run(decoder_outputs.get('decoder_outputs_3'),
                           feed_dict=feed_dict)
            out = (out[t] - 1)/9.0*(data_max[:num_states] - \
                   data_min[:num_states]) + data_min[:num_states]
            print('comparison of pred. and true K-values: %s vs %s'
                  % (out[0,2], ks[t]))
            
            # get alpha_norm
            #   -> alpha = alpha_norm*K_u_std + K_u_mean
            #   -> alpha_inp = (alpha-data_mean)/data_std
            alpha_norm = np.vstack(sess.run([layer[-1]],
                             feed_dict={K: np.reshape(out[0,2],[1,1])}))
            
            alpha = alpha_norm*K_u_std[1:5]+K_u_mean[1:5]
            alpha_inp = (alpha-data_mean[num_states:])/data_std[num_states:]
            print(alpha)
            print(alpha_inp)
            sess.run(control_input[t].assign(np.reshape(alpha_inp,
                                                       [1,num_wave_plates])))
                
        step = 0    
        true_obj = np.zeros([steps_phase_crtl,1])
        test_err = 1000
        extra_round = 0
        
        dec_out = np.vstack(sess.run(decoder_outputs.get('decoder_outputs_3'),
                            feed_dict=feed_dict))
        dec_out = (dec_out - 1)/9.0*(data_max[:num_states] - \
                   data_min[:num_states]) + data_min[:num_states]
        pred_obj = np.mean(dec_out[:,0]/dec_out[:,1])
        print('Pred. obj. fct. before optimization for K number %s in iteration %s: %s' % 
             (num, step, pred_obj))
        
        while (step < max_steps and (np.mean(true_obj) < 0.2 or test_err > 1e-1)):
            print('extra round %s' % extra_round)
            # Record summaries and test-set accuracy
            if test_err < 5e-2:
                runs = 5
            else:
                runs = 1
                
            for run in range(runs):
                sess.run([merged.get('merged_3'),
                          train_step.get('train_step_3')], feed_dict)
                
            summary,_ = sess.run([merged.get('merged_3'),
                                  error.get('error_3')], feed_dict)
            train_writer_cnt.add_summary(summary, step)
            
            step +=1
            
        dec_out = np.vstack(sess.run(decoder_outputs.get('decoder_outputs_3'),
                            feed_dict=feed_dict))
        dec_out = (dec_out - 1)/9.0*(data_max[:num_states] - \
                   data_min[:num_states]) + data_min[:num_states]
        pred_obj = np.mean(dec_out[:,0]/dec_out[:,1])
        print('Pred. obj. fct. for K number %s in iteration %s: %s' % 
             (num, step, pred_obj))
            
        # Update after identifying the angles
        laser_states = []
        angles = sess.run(control_input)
            
        #K_sim = ks * K_steps_std + K_steps_mean
            
        print('########################')
        print(step, K_simulation)
        print('########################')
             
        def prep_laser_states(laser_states):        
            laser_states = np.array(laser_states)
            
            # use the predicted birefringence coefficient instead of the true
            # value, as we cannot measure the birefengence value
            print('True obj. fct. for K number %s in iteration %s: %s' % 
                  (num, step, np.mean(laser_states[:,0]/laser_states[:,1])))
                
            
            z_vae = vae.transform(laser_states)
            K_vae = (z_vae - z_mu_mean)/z_mu_std
            #for i in range(steps_phase_crtl):
            #    if abs(K_vae[i,k_VAE_index] - ks[i]) > 5:
            #        print('Adjust K-value, diff = %s' % abs(K_vae[i,k_VAE_index] - ks[i]))
            #        K_vae[i,k_VAE_index] = ks[i]
            
            print(laser_states[:,:num_states-1])
            print(np.reshape(ks,[steps_phase_crtl,1])) # K_vae[:,k_VAE_index]
            print(laser_states[:,num_states-1:])
            
            laser_states_true = np.concatenate((laser_states[:,:num_states-num_latent_var],
                                np.reshape(ks,[steps_phase_crtl,1]),
                                laser_states[:,num_states-1:]),axis=1)
            # ks instead of K_vae[:,k_VAE_index] 
                
            laser_states_norm = np.zeros([steps_phase_crtl,1,num_parameters])
            for i in range(num_parameters):
                laser_states_norm[:,:,i] = np.reshape((laser_states[:,i] - \
                                 data_mean_ctrl[i])/data_std_ctrl[i],[steps_phase_crtl,1])
                 
            """
            next_iter = (0: batch_v_p_hist[10:,:,:],    1: batch_v_p_cur[10:,:,:],
                         2: batch_v_hist,               3: batch_v_cur,
                         4: batch_u_comp_h,             5: batch_u_comp_c,
                         6: batch_u_hist,               7: batch_u_cur)
            """
            
            laser_states_norm_K = 1 + 9*(laser_states_true-data_min)/(data_max-data_min)
            feed_dict.update({x_true.get('x_true_3')[t]: 
                np.reshape(laser_states_norm_K[t,:num_states], [1, num_states])
                for t in range(steps_phase_crtl)})
                 
            summary, test_err = sess.run([merged.get('merged_4'),
                                             error.get('error_4')],
                                             #train_step.get('train_step_4')], 
                                             feed_dict=feed_dict)
            train_writer_cnt_2.add_summary(summary, step)
            print('Error in step %s: %s' % (step, test_err))
            
            return (laser_states_norm, laser_states_norm_K, laser_states_true, 
                    test_err, K_vae)
        
        res_past = 0.0
        
        # Denormalization of the angles
        for angle, t in zip(angles,range(steps_phase_crtl)):
            ang = []
            def run_sim(angle, K_simu):
                for a in range(np.size(angle)):
                    # alpha = alpha_inp*data_std + data_mean
                    angle[0][a] = angle[0][a]*data_std[a+num_states] + \
                                  data_mean[a+num_states]
                    #angle[0][a] = angle[0][a]*K_u_std[a+1] + \
                    #              K_u_mean[a+1]
                    ang.append(angle[0][a])
                    
                # Run simulations using identified angles
                (_,states) = laser_simulation(uvt, ang[0], ang[1], ang[2], 
                                              ang[3], K_simu)
                
                return states
            
            states = run_sim(angle, K_simulation[t])
            
            res_current = states[0]/states[1]
            
            if (t > 0 and res_past - res_current > 0.015):
                angle = angles[t-1]
                states_temp = run_sim(angle, K_simulation[t])
                
                res_new = states_temp[0]/states_temp[1]
                
                if res_new > res_current:
                    states = states_temp
                    res_past = res_new
                    
            if ((t == 1 or t == steps_phase_crtl - 1) 
              and res_current - res_past > 0.015):
                states_temp = run_sim(angle, K_simulation[t-1])
                
                res_0 = states_temp[0]/states_temp[1]
                
                if res_0 > res_past:
                    laser_states[0] = states_temp
                    
            z_vae = vae.transform(np.reshape(states,[1,num_parameters]))
            K_vae = (z_vae - z_mu_mean)/z_mu_std
            
            print(K_vae[0,k_VAE_index], dec_out[t,2], ks[t])
                
            laser_states.append(states)
            
            res_past = res_current
        
        (laser_states_norm, laser_states_norm_K, laser_states_true, test_err,
         K_vae) = prep_laser_states(laser_states)
        
        if test_err > 1.25:
            print('Adjusting since error is too large')
            laser_states = []
            for t in range(steps_phase_crtl):
                alpha_norm = np.vstack(sess.run([layer[-1]],
                                feed_dict={K: np.reshape(ks[t],[1,1])}))[0]
                #ang = map_out*data_std[num_states:]+data_mean[num_states:]
                alpha = alpha_norm*K_u_std[1:5]+K_u_mean[1:5]
                
                alpha_inp = (alpha-data_mean[num_states:])/data_std[num_states:]
                sess.run(control_input[t].assign(np.reshape(alpha_inp,
                                                       [1,num_wave_plates])))
                
                (_,states) = laser_simulation(uvt,alpha[0], alpha[1], alpha[2], 
                                              alpha[3], K_simulation[t])
                
                res_current = states[0]/states[1]
            
                if (t > 0 and res_past - res_current > 0.015):
                    print('Res_past: %s, Res_current: %s, diff = %s' % (res_past, res_current, res_past - res_current))
                    angle = alpha_past.copy()
                    (_, states_temp) = laser_simulation(uvt,angle[0], angle[1],
                                                        angle[2], angle[3], 
                                                        K_simulation[t])
                    
                    res_new = states_temp[0]/states_temp[1]
                    
                    if res_new > res_current:
                        print('Res_new: %s, Res_current: %s' % (res_new, res_current))
                        states = states_temp
                        res_past = res_new
                        alpha = alpha_past.copy()
                        
                if ((t == 1 or t == steps_phase_crtl - 1) 
                  and res_current - res_past > 0.015):
                    print('Res_past: %s, Res_current: %s, diff = %s' % (res_past, res_current, res_past - res_current))
                    (_,states_temp) = laser_simulation(uvt,alpha[0], alpha[1], 
                                                       alpha[2], alpha[3], 
                                                       K_simulation[t-1])
                    
                    res_0 = states_temp[0]/states_temp[1]
                    
                    if res_0 > res_past:
                        print('Res_past: %s, Res_0: %s' % (res_past, res_0))
                        laser_states[0] = states_temp
                        
                alpha_past = alpha.copy()
                    
                laser_states.append(states)
                
                if t + 1 < steps_phase_crtl:
                    z_vae = vae.transform(np.reshape(states,[1,num_parameters]))
                    K_vae[t+1,:] = (z_vae - z_mu_mean)/z_mu_std
                
            (laser_states_norm, laser_states_norm_K, laser_states_true, _, 
             K_vae) = prep_laser_states(laser_states)
            
            dec_out = np.vstack(sess.run(decoder_outputs.get('decoder_outputs_3'),
                            feed_dict=feed_dict))
            dec_out = (dec_out - 1)/9.0*(data_max[:num_states] - \
                      data_min[:num_states]) + data_min[:num_states]
            
            
        step += 1
        
        if not pre_learning:
            # Identify feed_dict for the next steps_phase_crtl angles
            inp = np.zeros([2*steps_phase_crtl,1,num_inputs_RNN])
            inp[:,:,:] = inp_ctrl[:,:,:]
            inp[steps_phase_crtl+1:,:,
                (2*delay+1)*num_parameters:(2*delay+2)*num_parameters] = \
                laser_states_norm[:steps_phase_crtl-1,:,:]
            inp[steps_phase_crtl:,:,
                num_inputs_RNN-num_wave_plates:num_inputs_RNN] = \
            laser_states_norm[:,:,num_states - num_latent_var:]
            for i in range(steps_phase_crtl):
                if i > 0:
                    if i < 7:
                        inp[i+steps_phase_crtl:,:, (2*delay+2)*num_parameters + \
                            (delay-i)*num_wave_plates:(2*delay+2)*num_parameters + \
                            (delay + 1 - i)*num_wave_plates] = \
                        laser_states_norm[:steps_phase_crtl-i,:,num_states - num_latent_var:]        
                if i > 1:
                    inp[i+steps_phase_crtl:,:,
                        (2*delay+2-i)*num_parameters:(2*delay+3-i)*num_parameters] = \
                        laser_states_norm[:steps_phase_crtl-i,:,:]
                
            inp_ctrl = np.zeros([2*steps_phase_crtl,1,num_inputs_RNN])
            inp_ctrl[:steps_phase_crtl,:,:] = inp[steps_phase_crtl:,:,:]
            shift = np.concatenate((inp[2*steps_phase_crtl-1,:,:12*num_parameters],
                                    laser_states_norm[steps_phase_crtl-1,:,:]),
            axis=1)
            for i in range(steps_phase_crtl):
                inp_ctrl[i+steps_phase_crtl,:,:(12-i)*num_parameters] = shift[
                        :,num_parameters*(i+1):(13)*num_parameters]
                if i < (delay + 1):
                    inp_ctrl[i+steps_phase_crtl,:,
                          12*num_parameters:(12*num_parameters + \
                                             (delay - i)*num_wave_plates)] = \
                    inp[2*steps_phase_crtl-1,:,
                    num_inputs_RNN-(delay - i)*num_wave_plates:num_inputs_RNN]
                
            track.append(inp_ctrl)
                
            feed_dict = {}
            feed_dict = {encoder_inp.get(
                'encoder_inp_3')[t]: inp_ctrl[t,:,:] for t in range(steps_phase_crtl)}
            feed_dict.update({decoder_inp.get(
                'decoder_inp_3')[t]: inp_ctrl[t+steps_phase_crtl,:,:] for t in range(
                        steps_phase_crtl)})
            feed_dict.update({keep_prob.get('keep_prob_3'): 1})
            feed_dict.update({x_true.get(
                'x_true_3')[t]: np.array(np.reshape(laser_states_norm_K,
                    [steps_phase_crtl,1,num_parameters + num_latent_var]))[t,:,:num_states]
                    for t in range(steps_phase_crtl)})
    
        """     
        print('Step Four')
        Ks_comp.append([K_vae, dec_out[:,2]])
        if (np.mean(laser_states[:,0]/laser_states[:,1]) > 0.19):
            print('K_u_mapping: Angles = %s' % laser_states[0,num_states - num_latent_var:])
            for t in range(steps_phase_crtl):
                summary,_ = sess.run([merged.get('merged_5'),
                                      train_step.get('train_step_5')],
                    feed_dict={K: np.reshape(K_vae[t,k_VAE_index],[1,1]),
                               x_true['x_true_5']:np.reshape(
                                       laser_states_true[t,num_states:],
                                       [1, num_wave_plates])})
                train_writer_3.add_summary(summary, step)
                
            
        #saver.save(sess, os.path.join(os.getcwd(),'Save/trained_vars'),
        #                write_meta_graph=False, global_step=n)
        """  
        if num == 0:
            with open('laser_states.csv', 'w',newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                pred_fct = dec_out[:,0]/dec_out[:,1]
                for state, time in zip(laser_states_true, range(len(laser_states_true))):
                    state = state.reshape([num_parameters + num_latent_var,]).tolist()
                    obj_fct = state[0]/state[1]
                    state.append(obj_fct)
                    state.append(pred_fct[time])
                    spamwriter.writerow(state)
            
        else:
            with open('laser_states.csv', 'a',newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                pred_fct = dec_out[:,0]/dec_out[:,1]
                for state, time in zip(laser_states_true, range(len(laser_states_true))):
                    state = state.reshape([num_parameters + num_latent_var,]).tolist()
                    obj_fct = state[0]/state[1]
                    state.append(obj_fct)
                    state.append(pred_fct[time])
                    spamwriter.writerow(state)
        

#%%
#with tf.Session(config=config) as sess:
a = True
if a:
    sess = tf.Session(config=config) 

    sess.run(tf.global_variables_initializer())
    
    if reuse_model:
        train_writer_cnt = tf.summary.FileWriter(FLAGS.log_dir + '/train_cnt', 
                                                 sess.graph)
        train_writer_cnt_2 = tf.summary.FileWriter(FLAGS.log_dir + '/train_cnt_2')
        train_writer_3 = tf.summary.FileWriter(FLAGS.log_dir + '/train_3')
        test_writer_3 = tf.summary.FileWriter(FLAGS.log_dir + '/test_3')
    
    print(sess.run(W_lc_current))
    
    saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),
                                                               './Save/')))
    #print(sess.run(W_lc_current))
    saver_map.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),
                                                            './Save_map/')))
    saver_vae.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(),
                                                           './Save_vae/')))
    print(sess.run(W_lc_current))
    
    # delete afterwards
    #sess.run(mean_Energy.assign(data_mean[0]))
    #saver.save(sess, os.path.join(os.getcwd(),'Save/trained_vars'),
    #                            write_meta_graph=False, global_step=0)
    """
    T = 60
    n = 256
    t2 = np.linspace(-T/2,T/2,n+1)
    t_dis = t2[0:n].reshape([1,n])      # time discretization
    u=np.reshape(sech(t_dis/2), [n,])   # orthogonally polarized electric field 
    v=np.reshape(sech(t_dis/2), [n,])   # envelopes in the optical fiber
    ut=np.fft.fft(u).reshape(n,)        # fast fourier transformation of the
    vt=np.fft.fft(v).reshape(n,)        # electrical fields
    uvt=np.concatenate([ut, vt], axis=0)# concatenation of ut and vt
    
    init_states = np.zeros([2*steps_phase_crtl,1,
                     (2*delay + 2 + steps_phase_crtl)*num_parameters])
    init_angles = np.zeros([2*steps_phase_crtl,1,
                     (delay + 1 + steps_phase_crtl)*num_wave_plates])
    for t in range(22):
        print(K_steps[t])
        map_out = sess.run([layer[-1]],
                               feed_dict={K: np.reshape(K_steps[t],[1,1])})
            
        print(map_out[0][0][0], map_out[0][0][1], map_out[0][0][2], map_out[0][0][3])
            
        (_,states) = laser_simulation(uvt, map_out[0][0][0], 
                                          map_out[0][0][1], map_out[0][0][2],
                                          map_out[0][0][3], K_sim[t])
        print(states)
        
        z_vae = vae.transform(np.reshape(states,[1,num_parameters-1]))
        K_vae = (z_vae - z_mu_mean)/z_mu_std
        print(K_vae[:,k_VAE_index], d_dataset[t,2])
        
        states = (states - data_mean_ctrl) / data_std_ctrl
        states = np.concatenate((states[:num_states-1],
                                     K_vae[:,k_VAE_index],
                                     states[num_states-1:]),axis=0)
        
        init_states[0,0,t*num_parameters:(t+1)*num_parameters] = states
        if t > delay + 1:
                init_angles[0,0,(t-delay-2)*num_wave_plates:(t-delay-1)*num_wave_plates] = \
                states[num_states:]
                
    for t in range(1,2*steps_phase_crtl):
            init_states[t,0,0:(2*delay + 2 + steps_phase_crtl - t)*num_parameters] = \
            init_states[0,0,num_parameters*t:(2*delay + 2 + steps_phase_crtl)*num_parameters]
            if t < (delay + 1 + steps_phase_crtl):
                init_angles[t,0,0:(delay + 1 + steps_phase_crtl - t)*num_wave_plates] = \
                init_angles[0,0,t*num_wave_plates:(delay + 1 + steps_phase_crtl)*num_wave_plates]
         
    inp_ctrl = np.concatenate((init_states[:,:,:((2*delay + 2)*num_parameters)],
                                   init_angles[:,:,:((delay + 1)*num_wave_plates)]),
                                   axis=2)
    feed_dict = {}
    feed_dict = {encoder_inp.get(
            'encoder_inp_3')[t]: inp_ctrl[t,:,:] for t in range(steps_phase_crtl)}
    feed_dict.update({decoder_inp.get(
            'decoder_inp_3')[t]: inp_ctrl[t+steps_phase_crtl,:,:] for t in range(
                    steps_phase_crtl)})
    feed_dict.update({keep_prob.get('keep_prob_3'): 1})
    feed_dict.update({x_true.get('x_true_3')[t]: np.array([[0,0,0]])
                                     for t in range(steps_phase_crtl)})
        
    sess.run(control_input[0].assign(np.reshape(inp_ctrl[9,0,104:108],[1,4])))
    dec_out = np.vstack(sess.run(decoder_outputs.get('decoder_outputs_3'),
                            feed_dict=feed_dict))
    dec_out = (dec_out - data_mean[:3])/data_std[:3]
    print(data[:22,:3])
    print('Inp: %s' % inp_ctrl[:12,:,77:80])
    print('DEC_OUT: %s' % dec_out)
    
    inp_p_past = sess.run(obs_p_past_1,feed_dict=feed_dict)
    inp_p_cur = sess.run(obs_p_cur_1,feed_dict=feed_dict)
    inp_c_past = sess.run(obs_c_past_1,feed_dict=feed_dict)
    inp_c_cur = sess.run(obs_c_cur_1,feed_dict=feed_dict)
    inp_f_past = sess.run(obs_f_past_1,feed_dict=feed_dict)
    inp_f_cur = sess.run(obs_f_cur_1,feed_dict=feed_dict)
    
    inp_c_past = np.vstack(inp_c_past)
    inp_p_past = np.vstack(inp_p_past)
    inp_f_past = np.vstack(inp_f_past)
    inp_f_cur = np.vstack(inp_f_cur)
    inp_c_cur = np.vstack(inp_c_cur)
    inp_p_cur = np.vstack(inp_p_cur)
    inp = np.concatenate((inp_p_past,inp_p_cur,inp_c_past,inp_c_cur,inp_f_past,inp_f_cur),axis=1)
    
    print(inp[:,79])
    
    feed_dict,_,_,_ = feed_inp(2, steps_phase_2,train_batch_size, True)
    dec_out = np.vstack(sess.run(decoder_outputs.get('decoder_outputs_2'),
                            feed_dict=feed_dict))
    print(dec_out[0,:])
    """
    
    control_pred(K_steps, pre_learning=False)
    
    train_writer_cnt.close()
    train_writer_cnt_2.close()
    train_writer_3.close()
    test_writer_3.close()
    
