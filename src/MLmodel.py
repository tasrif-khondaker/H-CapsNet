import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
    #system
import os
import sys
import csv

import math
import random
import matplotlib
import matplotlib.pyplot as plt

def squash(s, axis=-1, name="squash"):
    """
    non-linear squashing function to manupulate the length of the capsule vectors
    :param s: input tensor containing capsule vectors
    :param axis: If `axis` is a Python integer, the input is considered a batch of vectors,
      and `axis` determines the axis in `tensor` over which to compute squash.
    :return: a Tensor with same shape as input vectors
    """
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                         keepdims=True)
        safe_norm = tf.sqrt(squared_norm + keras.backend.epsilon())
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
        
def safe_norm(s, axis=-1, keepdims=False, name="safe_norm"):
    """
    Safe computation of vector 2-norm
    :param s: input tensor
    :param axis: If `axis` is a Python integer, the input is considered a batch 
      of vectors, and `axis` determines the axis in `tensor` over which to 
      compute vector norms.
    :param keepdims: If True, the axis indicated in `axis` are kept with size 1.
      Otherwise, the dimensions in `axis` are removed from the output shape.
    :param name: The name of the op.
    :return: A `Tensor` of the same type as tensor, containing the vector norms. 
      If `keepdims` is True then the rank of output is equal to
      the rank of `tensor`. If `axis` is an integer, the rank of `output` is 
      one less than the rank of `tensor`.
    """
    with tf.name_scope(name):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                 keepdims=keepdims)
        return tf.sqrt(squared_norm + keras.backend.epsilon())    
        
class SecondaryCapsule(keras.layers.Layer):
    """
    The Secondary Capsule layer With Dynamic Routing Algorithm. 
    input shape = [None, input_num_capsule, input_dim_capsule] 
    output shape = [None, num_capsule, dim_capsule]
    :param n_caps: number of capsules in this layer
    :param n_dims: dimension of the output vectors of the capsules in this layer
    """
    def __init__(self, n_caps, n_dims, routings=2, **kwargs):
        super().__init__(**kwargs)
        self.n_caps = n_caps
        self.n_dims = n_dims
        self.routings = routings
    def build(self, batch_input_shape):
        # transformation matrix
        self.W = self.add_weight(
            name="W", 
            shape=(1, batch_input_shape[1], self.n_caps, self.n_dims, batch_input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.1))
        super().build(batch_input_shape)
    def call(self, X):
        # predict output vector
        batch_size = tf.shape(X)[0]
        caps1_n_caps = tf.shape(X)[1] 
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        caps1_output_expanded = tf.expand_dims(X, -1, name="caps1_output_expanded")
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, self.n_caps, 1, 1], name="caps1_output_tiled")
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")
        
        # rounting by agreement
        # routing weights
        raw_weights = tf.zeros([batch_size, caps1_n_caps, self.n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            routing_weights = tf.nn.softmax(raw_weights, axis=2, name="routing_weights")
            weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name="weighted_sum")
            caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_")
            caps2_output_squeezed = tf.squeeze(caps2_output_round_1, axis=[1,4], name="caps2_output_squeezed")
            if i < self.routings - 1:
                caps2_output_round_1_tiled = tf.tile(
                                        caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
                                        name="caps2_output_tiled_round_")
                agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")
                raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_")
                raw_weights = raw_weights_round_2
        return caps2_output_squeezed
    def compute_output_shape(self, batch_input_shape):
        return (batch_input_shape[0], self.n_caps, self.n_dims)
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "n_caps": self.n_caps, 
                "n_dims": self.n_dims,
                "routings": self.routings}
                
class LengthLayer(keras.layers.Layer):
    """
    Compute the length of capsule vectors.
    inputs: shape=[None, num_capsule, dim_vector]
    output: shape=[None, num_capsule]
    """
    def call(self, X):
        y_proba = safe_norm(X, axis=-1, name="y_proba")
        return y_proba
    def compute_output_shape(self, batch_input_shape): # in case the layer modifies the shape of its input, 
                                                        #you should specify here the shape transformation logic.
                                                        #This allows Keras to do automatic shape inference.
        return (batch_input_shape[0], batch_input_shape[1])
        
class Mask(keras.layers.Layer):
    """
    Mask a Tensor with the label during training 
    and mask a Tensor with predicted lable during test/inference
    input shapes
      X shape = [None, num_capsule, dim_vector] 
      y_true shape = [None, num_capsule] 
      y_pred shape = [None, num_capsule]
    output shape = [None, num_capsule * dim_vector]
    """
    def build(self, batch_input_shape):
        self.n_caps = batch_input_shape[0][1]
        self.n_dims = batch_input_shape[0][2]
        super().build(batch_input_shape)
    def call(self, input, training=None):
        X, y_true, y_proba = input
        if training:
            reconstruction_mask = y_true
        else:
            y_proba_argmax = tf.math.argmax(y_proba, axis=1, name="y_proba")
            y_pred = tf.one_hot(y_proba_argmax, depth=self.n_caps, name="y_pred")
            reconstruction_mask = y_pred
        reconstruction_mask_reshaped = tf.reshape(
                                      reconstruction_mask, [-1, self.n_caps, 1],
                                      name="reconstruction_mask_reshaped")
        caps2_output_masked = tf.multiply(
                            X, reconstruction_mask_reshaped,
                            name="caps2_output_masked")
        decoder_input = tf.reshape(caps2_output_masked,
                        [-1, self.n_caps * self.n_dims],
                        name="decoder_input")
        return decoder_input

    def compute_output_shape(self, input_shape):
        return (None, input_shape[0][1] * input_shape[0][2])
        
class MarginLoss(keras.losses.Loss):
    """
    Compute margin loss.
    y_true shape [None, n_classes] 
    y_pred shape [None, num_capsule] = [None, n_classes]
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5, **kwargs):
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
        super().__init__(**kwargs)
        
    def call(self, y_true, y_proba):
        present_error_raw = tf.square(tf.maximum(0., self.m_plus - y_proba), name="present_error_raw")
        absent_error_raw = tf.square(tf.maximum(0., y_proba - self.m_minus), name="absent_error_raw")
        L = tf.add(y_true * present_error_raw, self.lambda_ * (1.0 - y_true) * absent_error_raw,
           name="L")
        total_marginloss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        return total_marginloss
    
    def get_config(self): ### custom layers to be serializable as part of a Functional model
        base_config = super().get_config()
        return {**base_config, 
                "m_plus": self.m_plus,
                "m_minus": self.m_minus,
                "lambda_": self.lambda_}

                
def HCapsNet_2_Level_MNIST(input_shape,
                            no_coarse_class, no_fine_class,
                            PCap_n_dims = 8, SCap_n_dims = 16,
                            n_hidden1 = 512, n_hidden2 = 1024,
                            model_name : str = 'H-CapsNet'):
    """
    #H-Capsnet model for MNIST Dataset
    #Model has two hierarchical levels: It requires labels inputs for all the levels requires.
    #Number of classes can vary for the levels. It will impact the learning parameters.
    :input_Shape: Input shape of input images
    :no_coarse_class: Number of coarse class labels
    :no_fine_class: Number of coarse fine labels
    #Returns
    :model: H-CapsNet model for MNIST Dataset
    """
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv2')(conv1)

    conv2 = keras.layers.BatchNormalization()(conv2)

    #Convolution layer for coarse

    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc11.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc11)
    squashc1 = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    convc32 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    convc33 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3, 
                                  activation=tf.nn.relu,
                                  name='convc33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc33.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="SecondaryCapsule_fine")(squashcf)


    # Layer 4.3: Secondary capsule layer with routing by agreement
    # input [batch_size, 1152, 8], output [batch_size, 2, 16]
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="SecondaryCapsule_Coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    # input [batch_size, 10, 16], output [batch_size, 10]
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)


    # Layer 5.2: Compute the length of each capsule vector
    # input [batch_size, 10, 16], output [batch_size, 2]
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse_input = keras.Input(shape=(no_coarse_class,), name="coarse_image_label")

    # Mask layer
    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer]) 
    decoder_input_coarse = Mask(name='Mask_input_coarse')([SecondaryCapsule_coarse, coarse_input, coarse_pred_layer]) 

    # Decoder_Coarse
    # input [batch_size, 80], output [batch_size, 28, 28, 1]
   
    n_output = np.prod(input_shape)
  
    decoder_coarse = keras.models.Sequential(name='Decoder_coarse')
    decoder_coarse.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse'))

    # Decoder_fine
    # input [batch_size, 160], output [batch_size, 28, 28, 1]
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    coarse_decoder= decoder_coarse(decoder_input_coarse)
    fine_decoder= decoder_fine(decoder_input_fine)

    concatted = keras.layers.concatenate([fine_decoder, coarse_decoder])
    concatted = keras.layers.Dense(4, activation='relu')(concatted)
    Output_Layer = keras.layers.Dense(1, activation='linear',name='Final_output')(concatted)
    
    # Capsnet model
    model = keras.Model(inputs = [x_input, coarse_input, fine_input],
                        outputs = [coarse_pred_layer, fine_pred_layer, Output_Layer],
                        name = model_name)
    
    return model
    
def HCapsNet_2_Level_EMNIST(input_shape,
                            no_coarse_class,
                            no_fine_class,
                            PCap_n_dims = 8,SCap_n_dims = 16,
                            n_hidden1 = 512,n_hidden2 = 1024):
    """
    #H-Capsnet model for EMNIST Dataset
    #Model has two hierarchical levels: It requires labels inputs for all the levels requires.
    #Number of classes can vary for the levels. It will impact the learning parameters.
    :input_Shape: Input shape of input images
    :no_coarse_class: Number of coarse class labels
    :no_fine_class: Number of coarse fine labels
    #Returns
    :model: H-CapsNet model for EMNIST Dataset
    """
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv2')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)

    #Convolution layer for coarse
    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc11.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc11)
    squashc = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    convc32 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    convc33 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3, 
                                  activation=tf.nn.relu,
                                  name='convc33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc33.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Coarse")(squashc)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Fine")(squashcf)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_input")
    coarse_input = keras.Input(shape=(no_coarse_class,), name="coarse_input")

    # Mask layer

    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer]) 
    decoder_input_coarse = Mask(name='Mask_input_coarse')([SecondaryCapsule_coarse, coarse_input, coarse_pred_layer])
    
    n_output = np.prod(input_shape)

    # Decoder_Coarse
    decoder_coarse = keras.models.Sequential(name='Decoder_coarse')
    decoder_coarse.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse'))

    # Decoder_fine
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    coarse_decoder= decoder_coarse(decoder_input_coarse)
    fine_decoder= decoder_fine(decoder_input_fine)

    concatted = keras.layers.concatenate([fine_decoder, coarse_decoder])
    concatted = keras.layers.Dense(4, activation='relu')(concatted)
    Output_Layer = keras.layers.Dense(1, activation='linear',name='Final_output')(concatted)
    
    # Capsnet model
    model = keras.Model(
        inputs= [x_input, coarse_input, fine_input],
        outputs= [coarse_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet')
    
    return model

def HCapsNet_3_Level_FMNIST(input_shape,
                            no_coarse_class,no_medium_class,no_fine_class,
                            PCap_n_dims = 8,SCap_n_dims = 16,
                            n_hidden1 = 512,n_hidden2 = 1024):
    """
    #H-Capsnet model for Fashion-MNIST Dataset
    #Model has three hierarchical levels: It requires labels inputs for all the levels requires.
    #Number of classes can vary for the levels. It will impact the learning parameters.
    :input_Shape: Input shape of input images
    :no_coarse_class: Number of coarse class labels
    :no_medium_class: Number of medium class labels
    :no_fine_class: Number of coarse fine labels
    #Returns
    :model: H-CapsNet model for Fashion-MNIST Dataset
    """    
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv2')(conv1)

    conv2 = keras.layers.BatchNormalization()(conv2)

    #Convolution layer for coarse
    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc11.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc11)
    squashc1 = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for Medium
    convc21 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc21')(conv2)
    convc21 = keras.layers.BatchNormalization()(convc21)

    convc22 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc22')(convc21)
    convc22 = keras.layers.BatchNormalization()(convc22)

    convc23 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc23')(convc22)
    convc23 = keras.layers.BatchNormalization()(convc23)

    reshapec2 = keras.layers.Reshape((int((tf.reduce_prod(convc23.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c2")(convc23)
    squashc2 = keras.layers.Lambda(squash, name='squash_layer_c2')(reshapec2)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    convc32 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    convc33 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3, 
                                  activation=tf.nn.relu,
                                  name='convc33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc33.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_medium = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Medium")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    medium_pred_layer = LengthLayer(name='Medium_prediction_output_layer')(SecondaryCapsule_medium)

    # Layer 5.3: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse2_input = keras.Input(shape=(no_medium_class,), name="coarse2_image_label")
    coarse1_input = keras.Input(shape=(no_coarse_class,), name="coarse1_image_label")

    # Mask layer
    n_output = np.prod(input_shape)

    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_medium, coarse2_input, medium_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse, coarse1_input, coarse_pred_layer])

    # Decoder_fine
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    # Decoder_Coarse2
    decoder_coarse2 = keras.models.Sequential(name='Decoder_coarse2')
    decoder_coarse2.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_medium_class))
    decoder_coarse2.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse2.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse2.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse2'))

    # Decoder_Coarse1
    decoder_coarse1 = keras.models.Sequential(name='Decoder_coarse1')
    decoder_coarse1.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse1.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse1.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse1.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse1'))

    fine_decoder= decoder_fine(decoder_input_fine)
    coarse2_decoder= decoder_coarse2(decoder_input_coarse2)

    coarse1_decoder= decoder_coarse1(decoder_input_coarse1)

    concatted = keras.layers.concatenate([fine_decoder, coarse2_decoder, coarse1_decoder])

    concatted = keras.layers.Dense(4, activation='relu')(concatted)

    Output_Layer = keras.layers.Dense(3, activation='linear',name='Final_output')(concatted)
    
    # Capsnet model
    model = keras.Model(
        inputs= [x_input, coarse1_input, coarse2_input, fine_input],
        outputs= [coarse_pred_layer, medium_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet')
    
    return model
    
def HCapsNet_3_Level(input_shape,
                    no_coarse_class,no_medium_class,no_fine_class,
                    PCap_n_dims = 8,SCap_n_dims = 16,
                    n_hidden1 = 512,n_hidden2 = 1024):
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder_conv_1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder_conv_2')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    
    conv2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_1')(conv2)

    #Convolution layer for Coarse
    convc11 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)
    
    convc12 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv12')(convc11)
    convc12 = keras.layers.BatchNormalization()(convc12)
    
    convc12 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_2')(convc12)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc12.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc12)
    squashc1 = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for Medium
    
    convc21 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv21')(conv2)
    convc21 = keras.layers.BatchNormalization()(convc21)
    
    convc22 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv22')(convc21)
    convc22 = keras.layers.BatchNormalization()(convc22)
    
    convc22 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_12')(convc22)
    
    convc23 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv23')(convc22)
    convc23 = keras.layers.BatchNormalization()(convc23)
    
    convc24 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv24')(convc23)
    convc24 = keras.layers.BatchNormalization()(convc24)
    
    convc24 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpoo_22')(convc24)
    
    reshapec2 = keras.layers.Reshape((int((tf.reduce_prod(convc24.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c2")(convc24)
    squashc2 = keras.layers.Lambda(squash, name='squash_layer_c2')(reshapec2)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    
    convc32 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    
    convc32 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpoo_31')(convc32)
    
    convc33 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)
    
    convc34 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv34')(convc33)
    convc34 = keras.layers.BatchNormalization()(convc34)
    
    convc34 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpoo_32')(convc34)
        
    convc35 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='FE_conv35')(convc34)
    convc35 = keras.layers.BatchNormalization()(convc35)
        
    convc36 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='FE_conv36')(convc35)
    convc36 = keras.layers.BatchNormalization()(convc36)
    
    convc36 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_33')(convc36)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc36.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc36)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_medium = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Medium")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    medium_pred_layer = LengthLayer(name='Medium_prediction_output_layer')(SecondaryCapsule_medium)

    # Layer 5.3: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse2_input = keras.Input(shape=(no_medium_class,), name="coarse2_image_label")
    coarse1_input = keras.Input(shape=(no_coarse_class,), name="coarse1_image_label")

    # Mask layer
    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_medium, coarse2_input, medium_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse, coarse1_input, coarse_pred_layer])

    n_output = np.prod(input_shape)

    # Decoder_fine
    # input [batch_size, 160], output [batch_size, 32, 32, 32]
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    # Decoder_Coarse2
    # input [batch_size, 112], output [batch_size, 32, 32, 3]
    decoder_coarse2 = keras.models.Sequential(name='Decoder_coarse2')
    decoder_coarse2.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_medium_class))
    decoder_coarse2.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse2.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse2.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse2'))

    # Decoder_Coarse1
    # input [batch_size, 32], output [batch_size, 32, 32, 3]
    decoder_coarse1 = keras.models.Sequential(name='Decoder_coarse1')
    decoder_coarse1.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse1.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse1.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse1.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse1'))

    fine_decoder= decoder_fine(decoder_input_fine)
    coarse2_decoder= decoder_coarse2(decoder_input_coarse2)

    coarse1_decoder= decoder_coarse1(decoder_input_coarse1)

    concatted = keras.layers.concatenate([fine_decoder, coarse2_decoder, coarse1_decoder])

    concatted = keras.layers.Dense(4, activation='relu')(concatted)

    Output_Layer = keras.layers.Dense(3, activation='linear',name='Final_output')(concatted)

    # Capsnet model
    model = keras.Model(
        inputs= [x_input, coarse1_input, coarse2_input, fine_input],
        outputs= [coarse_pred_layer, medium_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet')
    
    return model

                
def HCapsNet_2_Level_MNIST_test(input_shape,
                            no_coarse_class, no_fine_class,
                            PCap_n_dims = 8, SCap_n_dims = 16,
                            n_hidden1 = 512, n_hidden2 = 1024,
                            model_name : str = 'H-CapsNet'):
    """
    #H-Capsnet model for MNIST Dataset
    #Model has two hierarchical levels: It requires labels inputs for all the levels requires.
    #Number of classes can vary for the levels. It will impact the learning parameters.
    :input_Shape: Input shape of input images
    :no_coarse_class: Number of coarse class labels
    :no_fine_class: Number of coarse fine labels
    #Returns
    :model: H-CapsNet model for MNIST Dataset
    """
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv2')(conv1)

    conv2 = keras.layers.BatchNormalization()(conv2)

    #Convolution layer for coarse

    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc11.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc11)
    squashc1 = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    convc32 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    convc33 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3, 
                                  activation=tf.nn.relu,
                                  name='convc33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc33.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="SecondaryCapsule_fine")(squashcf)


    # Layer 4.3: Secondary capsule layer with routing by agreement
    # input [batch_size, 1152, 8], output [batch_size, 2, 16]
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="SecondaryCapsule_Coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    # input [batch_size, 10, 16], output [batch_size, 10]
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)


    # Layer 5.2: Compute the length of each capsule vector
    # input [batch_size, 10, 16], output [batch_size, 2]
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    # Mask layer
    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_pred_layer, fine_pred_layer]) 
    decoder_input_coarse = Mask(name='Mask_input_coarse')([SecondaryCapsule_coarse, coarse_pred_layer, coarse_pred_layer]) 

    # Decoder_Coarse
    # input [batch_size, 80], output [batch_size, 28, 28, 1]
   
    n_output = np.prod(input_shape)
  
    decoder_coarse = keras.models.Sequential(name='Decoder_coarse')
    decoder_coarse.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse'))

    # Decoder_fine
    # input [batch_size, 160], output [batch_size, 28, 28, 1]
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    coarse_decoder= decoder_coarse(decoder_input_coarse)
    fine_decoder= decoder_fine(decoder_input_fine)

    concatted = keras.layers.concatenate([fine_decoder, coarse_decoder])
    concatted = keras.layers.Dense(4, activation='relu')(concatted)
    Output_Layer = keras.layers.Dense(1, activation='linear',name='Final_output')(concatted)
    
    # Capsnet model
    model = keras.Model(inputs = x_input,
                        outputs = [coarse_pred_layer, fine_pred_layer, Output_Layer],
                        name = model_name)
    
    return model
    
def HCapsNet_2_Level_EMNIST_test(input_shape,
                            no_coarse_class,
                            no_fine_class,
                            PCap_n_dims = 8,SCap_n_dims = 16,
                            n_hidden1 = 512,n_hidden2 = 1024):
    """
    #H-Capsnet model for EMNIST Dataset
    #Model has two hierarchical levels: It requires labels inputs for all the levels requires.
    #Number of classes can vary for the levels. It will impact the learning parameters.
    :input_Shape: Input shape of input images
    :no_coarse_class: Number of coarse class labels
    :no_fine_class: Number of coarse fine labels
    #Returns
    :model: H-CapsNet model for EMNIST Dataset
    """
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv2')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)

    #Convolution layer for coarse
    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc11.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc11)
    squashc = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    convc32 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    convc33 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3, 
                                  activation=tf.nn.relu,
                                  name='convc33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc33.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Coarse")(squashc)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Fine")(squashcf)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    # Mask layer

    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_pred_layer, fine_pred_layer]) 
    decoder_input_coarse = Mask(name='Mask_input_coarse')([SecondaryCapsule_coarse, coarse_pred_layer, coarse_pred_layer])
    
    n_output = np.prod(input_shape)

    # Decoder_Coarse
    decoder_coarse = keras.models.Sequential(name='Decoder_coarse')
    decoder_coarse.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse'))

    # Decoder_fine
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    coarse_decoder= decoder_coarse(decoder_input_coarse)
    fine_decoder= decoder_fine(decoder_input_fine)

    concatted = keras.layers.concatenate([fine_decoder, coarse_decoder])
    concatted = keras.layers.Dense(4, activation='relu')(concatted)
    Output_Layer = keras.layers.Dense(1, activation='linear',name='Final_output')(concatted)
    
    # Capsnet model
    model = keras.Model(
        inputs= x_input,
        outputs= [coarse_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet')
    
    return model

def HCapsNet_3_Level_FMNIST_test(input_shape,
                            no_coarse_class,no_medium_class,no_fine_class,
                            PCap_n_dims = 8,SCap_n_dims = 16,
                            n_hidden1 = 512,n_hidden2 = 1024):
    """
    #H-Capsnet model for Fashion-MNIST Dataset
    #Model has three hierarchical levels: It requires labels inputs for all the levels requires.
    #Number of classes can vary for the levels. It will impact the learning parameters.
    :input_Shape: Input shape of input images
    :no_coarse_class: Number of coarse class labels
    :no_medium_class: Number of medium class labels
    :no_fine_class: Number of coarse fine labels
    #Returns
    :model: H-CapsNet model for Fashion-MNIST Dataset
    """    
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(filters=32,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)

    conv2 = keras.layers.Conv2D(filters=64,
                                kernel_size=3,
                                padding="valid",
                                strides=1,
                                activation=tf.nn.relu,
                                name='conv2')(conv1)

    conv2 = keras.layers.BatchNormalization()(conv2)

    #Convolution layer for coarse
    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc11.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc11)
    squashc1 = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for Medium
    convc21 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc21')(conv2)
    convc21 = keras.layers.BatchNormalization()(convc21)

    convc22 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc22')(convc21)
    convc22 = keras.layers.BatchNormalization()(convc22)

    convc23 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc23')(convc22)
    convc23 = keras.layers.BatchNormalization()(convc23)

    reshapec2 = keras.layers.Reshape((int((tf.reduce_prod(convc23.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c2")(convc23)
    squashc2 = keras.layers.Lambda(squash, name='squash_layer_c2')(reshapec2)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(filters=128,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    convc32 = keras.layers.Conv2D(filters=256,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=1,
                                  activation=tf.nn.relu,
                                  name='convc32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    convc33 = keras.layers.Conv2D(filters=512,
                                  kernel_size=3,
                                  padding="valid",
                                  strides=3, 
                                  activation=tf.nn.relu,
                                  name='convc33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc33.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_medium = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Medium")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    medium_pred_layer = LengthLayer(name='Medium_prediction_output_layer')(SecondaryCapsule_medium)

    # Layer 5.3: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    # Mask layer
    n_output = np.prod(input_shape)

    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_pred_layer, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_medium, medium_pred_layer, medium_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse, coarse_pred_layer, coarse_pred_layer])

    # Decoder_fine
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    # Decoder_Coarse2
    decoder_coarse2 = keras.models.Sequential(name='Decoder_coarse2')
    decoder_coarse2.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_medium_class))
    decoder_coarse2.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse2.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse2.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse2'))

    # Decoder_Coarse1
    decoder_coarse1 = keras.models.Sequential(name='Decoder_coarse1')
    decoder_coarse1.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse1.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse1.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse1.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse1'))

    fine_decoder= decoder_fine(decoder_input_fine)
    coarse2_decoder= decoder_coarse2(decoder_input_coarse2)

    coarse1_decoder= decoder_coarse1(decoder_input_coarse1)

    concatted = keras.layers.concatenate([fine_decoder, coarse2_decoder, coarse1_decoder])

    concatted = keras.layers.Dense(4, activation='relu')(concatted)

    Output_Layer = keras.layers.Dense(3, activation='linear',name='Final_output')(concatted)
    
    # Capsnet model
    model = keras.Model(
        inputs= x_input,
        outputs= [coarse_pred_layer, medium_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet')
    
    return model
    
def HCapsNet_3_Level_test(input_shape,
                    no_coarse_class,no_medium_class,no_fine_class,
                    PCap_n_dims = 8,SCap_n_dims = 16,
                    n_hidden1 = 512,n_hidden2 = 1024):
    # Input image
    x_input = keras.layers.Input(shape=input_shape, name="Input_Image")
    # Encoder Layer
    conv1 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder_conv_1')(x_input)
    conv1 = keras.layers.BatchNormalization()(conv1)
    
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='encoder_conv_2')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    
    conv2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_1')(conv2)

    #Convolution layer for Coarse
    convc11 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)
    
    convc12 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv12')(convc11)
    convc12 = keras.layers.BatchNormalization()(convc12)
    
    convc12 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_2')(convc12)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((int((tf.reduce_prod(convc12.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c1")(convc12)
    squashc1 = keras.layers.Lambda(squash, name='squash_layer_c1')(reshapec1)

    #Convolution layer for Medium
    
    convc21 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv21')(conv2)
    convc21 = keras.layers.BatchNormalization()(convc21)
    
    convc22 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv22')(convc21)
    convc22 = keras.layers.BatchNormalization()(convc22)
    
    convc22 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_12')(convc22)
    
    convc23 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv23')(convc22)
    convc23 = keras.layers.BatchNormalization()(convc23)
    
    convc24 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv24')(convc23)
    convc24 = keras.layers.BatchNormalization()(convc24)
    
    convc24 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpoo_22')(convc24)
    
    reshapec2 = keras.layers.Reshape((int((tf.reduce_prod(convc24.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_c2")(convc24)
    squashc2 = keras.layers.Lambda(squash, name='squash_layer_c2')(reshapec2)

    #Convolution layer for fine
    convc31 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv31')(conv2)
    convc31 = keras.layers.BatchNormalization()(convc31)
    
    convc32 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='FE_conv32')(convc31)
    convc32 = keras.layers.BatchNormalization()(convc32)
    
    convc32 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpoo_31')(convc32)
    
    convc33 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv33')(convc32)
    convc33 = keras.layers.BatchNormalization()(convc33)
    
    convc34 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='FE_conv34')(convc33)
    convc34 = keras.layers.BatchNormalization()(convc34)
    
    convc34 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpoo_32')(convc34)
        
    convc35 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='FE_conv35')(convc34)
    convc35 = keras.layers.BatchNormalization()(convc35)
        
    convc36 = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='FE_conv36')(convc35)
    convc36 = keras.layers.BatchNormalization()(convc36)
    
    convc36 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_33')(convc36)

    reshapef = keras.layers.Reshape((int((tf.reduce_prod(convc36.shape[1:]).numpy())/PCap_n_dims), PCap_n_dims)
                                       , name="reshape_layer_f")(convc36)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_medium = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Medium")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Secondary_Caps_Coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    medium_pred_layer = LengthLayer(name='Medium_prediction_output_layer')(SecondaryCapsule_medium)

    # Layer 5.3: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)


    # Mask layer
    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_pred_layer, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_medium, medium_pred_layer, medium_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse, coarse_pred_layer, coarse_pred_layer])

    n_output = np.prod(input_shape)

    # Decoder_fine
    # input [batch_size, 160], output [batch_size, 32, 32, 32]
    decoder_fine = keras.models.Sequential(name='Decoder_fine')
    decoder_fine.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_fine_class))
    decoder_fine.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_fine.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_fine.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_fine'))

    # Decoder_Coarse2
    # input [batch_size, 112], output [batch_size, 32, 32, 3]
    decoder_coarse2 = keras.models.Sequential(name='Decoder_coarse2')
    decoder_coarse2.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_medium_class))
    decoder_coarse2.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse2.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse2.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse2'))

    # Decoder_Coarse1
    # input [batch_size, 32], output [batch_size, 32, 32, 3]
    decoder_coarse1 = keras.models.Sequential(name='Decoder_coarse1')
    decoder_coarse1.add(keras.layers.Dense(n_hidden1, activation='relu', input_dim=SCap_n_dims*no_coarse_class))
    decoder_coarse1.add(keras.layers.Dense(n_hidden2, activation='relu'))
    decoder_coarse1.add(keras.layers.Dense(n_output, activation='sigmoid'))
    decoder_coarse1.add(keras.layers.Reshape(target_shape=input_shape, name='recon_output_layer_coarse1'))

    fine_decoder= decoder_fine(decoder_input_fine)
    coarse2_decoder= decoder_coarse2(decoder_input_coarse2)

    coarse1_decoder= decoder_coarse1(decoder_input_coarse1)

    concatted = keras.layers.concatenate([fine_decoder, coarse2_decoder, coarse1_decoder])

    concatted = keras.layers.Dense(4, activation='relu')(concatted)

    Output_Layer = keras.layers.Dense(3, activation='linear',name='Final_output')(concatted)

    # Capsnet model
    model = keras.Model(
        inputs= x_input,
        outputs= [coarse_pred_layer, medium_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet')
    
    return model

def B_CNN_Model_B(input_shape, num_class_c, num_class_m, num_class_f, 
                  model_name:str='B_CNN_Model_B'):
    
    img_input = keras.layers.Input(shape=input_shape, name='input')

    #--- block 1 ---
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #--- block 2 ---
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #--- coarse 1 branch ---
    c_bch = keras.layers.Flatten(name='c1_flatten')(x)
    c_bch = keras.layers.Dense(256, activation='relu', name='c_fc_1')(c_bch)
    c_bch = keras.layers.BatchNormalization()(c_bch)
    c_bch = keras.layers.Dropout(0.5)(c_bch)
    c_bch = keras.layers.Dense(256, activation='relu', name='c_fc_2')(c_bch)
    c_bch = keras.layers.BatchNormalization()(c_bch)
    c_bch = keras.layers.Dropout(0.5)(c_bch)
    c_pred = keras.layers.Dense(num_class_c, activation='softmax', name='c_predictions')(c_bch)

    #--- block 3 ---
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #--- coarse 2 branch ---
    m_bch = keras.layers.Flatten(name='c2_flatten')(x)
    m_bch = keras.layers.Dense(512, activation='relu', name='m_fc_1')(m_bch)
    m_bch = keras.layers.BatchNormalization()(m_bch)
    m_bch = keras.layers.Dropout(0.5)(m_bch)
    m_bch = keras.layers.Dense(512, activation='relu', name='m_fc_2')(m_bch)
    m_bch = keras.layers.BatchNormalization()(m_bch)
    m_bch = keras.layers.Dropout(0.5)(m_bch)
    m_pred = keras.layers.Dense(num_class_m, activation='softmax', name='m_predictions')(m_bch)

    #--- block 4 ---
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #--- fine block ---
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(1024, activation='relu', name='f_fc_1')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(1024, activation='relu', name='f_fc2_2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.5)(x)
    f_pred = keras.layers.Dense(num_class_f, activation='softmax', name='f_predictions')(x)
    model = keras.Model(img_input, [c_pred, m_pred, f_pred], name=model_name)
    
    return model
    
def initial_lw(class_labels: dict):
    """
    Give dictionary input for hierarchical levels.
    Where the values for the input keys needs to be total number of classes in the levels.
    Example for 3 levels hierarchy with 2, 7, 10 class will be in following format:
    class_labels = {"coarse": coarse_class, "medium": medium_class,"fine": fine_class}
    :c:The Function will return initial loss weight values in a dictionary, based on number of classes in levels
    """

    q = {}
    for k, v in class_labels.items():
        q[k] = (1 - (v / sum(class_labels.values())))

    c = {}
    for k, v in class_labels.items():
        c[k] = (q[k] / sum(q.values()))
        
    return c
    
class LossWeightsModifier(keras.callbacks.Callback):
    
    def __init__(self, lossweight : dict,initial_lw : dict, directory : str):

        self.lossweight = lossweight
        self.directory = directory
        self.reconstruction_loss = lossweight['decoder_lw']
        
        if 'coarse_lw' in self.lossweight and 'medium_lw' in self.lossweight and 'fine_lw' in self.lossweight:
            self.coarse_lw = lossweight['coarse_lw']
            self.medium_lw = lossweight['medium_lw']
            self.fine_lw = lossweight['fine_lw']
        
            self.c1 = initial_lw['coarse'] # Initial LW for Coarse class
            self.c2 = initial_lw['medium'] # Initial LW for Fine class
            self.c3 = initial_lw['fine'] # Initial LW for Fine class
        
            self.header = ['Epoch',
                            'Coarse_Accuracy', 'Coarse_LossWeight',
                            'Medium_Accuracy', 'Medium_LossWeight',
                            'Fine_Accuracy', 'Fine_LossWeight']
            
        elif 'coarse_lw' in self.lossweight and 'fine_lw' in self.lossweight:

            self.coarse_lw = lossweight['coarse_lw']
            self.fine_lw = lossweight['fine_lw']
        
            self.c1 = initial_lw['coarse'] # Initial LW for Coarse class
            self.c2 = initial_lw['fine'] # Initial LW for Fine class
        
            self.header = ['Epoch',
                            'Coarse_Accuracy', 'Coarse_LossWeight',
                            'Fine_Accuracy', 'Fine_LossWeight']
          
        csv_file = open(self.directory+'/LossWeight.csv', 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(self.header)
        csv_file.close()
    
    def on_epoch_end(self, epoch, logs={}):
    
        if 'coarse_lw' in self.lossweight and 'medium_lw' in self.lossweight and 'fine_lw' in self.lossweight:
            # Taking Training Accuracy for Calculation
            ACC1 = logs.get('Coarse_prediction_output_layer_accuracy')
            ACC2 = logs.get('Medium_prediction_output_layer_accuracy')
            ACC3 = logs.get('Fine_prediction_output_layer_accuracy')
            
            # Taking Validation Accuracy just for printing
            VACC1 = logs.get('val_Coarse_prediction_output_layer_accuracy')
            VACC2 = logs.get('val_Medium_prediction_output_layer_accuracy')
            VACC3 = logs.get('val_Fine_prediction_output_layer_accuracy')
            
            #Calculating Tau Values for each classes
            tau1 = (1-ACC1) * self.c1
            tau2 = (1-ACC2) * self.c2
            tau3 = (1-ACC3) * self.c3
            
            # Updated Loss for each classes
            L1 = float((1-self.reconstruction_loss) * (tau1 / (tau1 + tau2 + tau3)))
            L2 = float((1-self.reconstruction_loss) * (tau2 / (tau1 + tau2 + tau3)))
            L3 = float((1-self.reconstruction_loss) * (tau3 / (tau1 + tau2 + tau3)))
            
            print('\033[91m','\033[1m',"\u2022",
                  "Coarse Accuracy =",'{:.2f}%'.format(ACC1*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC1*100),"|",
                  "LossWeight =",'{:.2f}'.format(L1),
                  '\033[0m')
            print('\033[91m','\033[1m',"\u2022",
                  "Medium Accuracy =",'{:.2f}%'.format(ACC2*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC2*100),"|",
                  "LossWeight =",'{:.2f}'.format(L2),
                  '\033[0m')
            print('\033[91m','\033[1m',"\u2022",
                  "Fine   Accuracy =",'{:.2f}%'.format(ACC3*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC3*100),"|",
                  "LossWeight =",'{:.2f}'.format(L3),
                  '\033[0m')
            
            ## Saving Data to CSV FILE##
            data = {'Epoch': epoch,
                    'Coarse_Accuracy': ACC1,
                    'Coarse_LossWeight': L1,
                    'Medium_Accuracy': ACC2,
                    'Medium_LossWeight': L2,
                    'Fine_Accuracy': ACC3,
                    'Fine_LossWeight': L3}
            
            with open(self.directory+'/LossWeight.csv', mode='a', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames = self.header)
                csv_writer.writerow(data)
                
            #Setting Loss weight Values
            K.set_value(self.coarse_lw, L1)
            K.set_value(self.medium_lw, L2)
            K.set_value(self.fine_lw, L3)
            
        elif 'coarse_lw' in self.lossweight and 'fine_lw' in self.lossweight:

            # Taking Training Accuracy for Calculation
            ACC1 = logs.get('Coarse_prediction_output_layer_accuracy')
            ACC2 = logs.get('Fine_prediction_output_layer_accuracy')
            
            # Taking Validation Accuracy just for printing
            VACC1 = logs.get('val_Coarse_prediction_output_layer_accuracy')
            VACC2 = logs.get('val_Fine_prediction_output_layer_accuracy')
            
            #Calculating Tau Values for each classes
            tau1 = (1-ACC1) * self.c1
            tau2 = (1-ACC2) * self.c2
            
            # Updated Loss for each classes
            L1 = float((1-self.reconstruction_loss) * (tau1 / (tau1 + tau2)))
            L2 = float((1-self.reconstruction_loss) * (tau2 / (tau1 + tau2)))
            
            print('\033[91m','\033[1m',"\u2022",
                  "Coarse Accuracy =",'{:.2f}%'.format(ACC1*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC1*100),"|",
                  "LossWeight =",'{:.2f}'.format(L1),
                  '\033[0m')
            print('\033[91m','\033[1m',"\u2022",
                  "Fine   Accuracy =",'{:.2f}%'.format(ACC2*100),"|",
                  "Val_Accuracy =",'{:.2f}%'.format(VACC2*100),"|",
                  "LossWeight =",'{:.2f}'.format(L2),
                  '\033[0m')
            
            ## Saving Data to CSV FILE##
            data = {'Epoch': epoch,
                    'Coarse_Accuracy': ACC1,
                    'Coarse_LossWeight': L1,
                    'Fine_Accuracy': ACC2,
                    'Fine_LossWeight': L2}
            
            with open(self.directory+'/LossWeight.csv', mode='a', newline='') as csv_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames = self.header)
                csv_writer.writerow(data)
            
            ## Setting Loss weight Values
            K.set_value(self.coarse_lw, L1)
            K.set_value(self.fine_lw, L2)
        
class model_analysis():
        def __init__(self, model: keras.Model, dataset: dict,batch_size=64):
            self.dataset = dataset
            self.model = model
            self.batch = batch_size
        
        def evaluate(self):
                       
            if 'y_test_coarse' in self.dataset and 'y_test_medium' in self.dataset and 'y_test_fine' in self.dataset:
                results = self.model.evaluate([self.dataset['x_test'],self.dataset['y_test_coarse'],self.dataset['y_test_medium'], self.dataset['y_test_fine']],[self.dataset['y_test_coarse'],self.dataset['y_test_medium'], self.dataset['y_test_fine'],self.dataset['x_test']],
                batch_size = self.batch,
                verbose=1)
                
                for n in range(len(results)):
                    print(str(n+1)+'.',self.model.metrics_names[n], '==>', results[n])
          
            elif 'y_test_coarse' in self.dataset and 'y_test_fine' in self.dataset:
                results = self.model.evaluate([self.dataset['x_test'],self.dataset['y_test_coarse'],self.dataset['y_test_fine']],[self.dataset['y_test_coarse'],self.dataset['y_test_fine'],self.dataset['x_test']],
                batch_size = self.batch,
                verbose=1)
                
                for n in range(len(results)):
                    print(str(n+1)+'.',self.model.metrics_names[n], '==>', results[n])
          
            return results
            
        def prediction(self):
            if 'y_test_coarse' in self.dataset and 'y_test_medium' in self.dataset and 'y_test_fine' in self.dataset:
                predictions = self.model.predict([self.dataset['x_test'],self.dataset['y_test_coarse'],self.dataset['y_test_medium'], self.dataset['y_test_fine']],batch_size = self.batch,verbose=1)
                
                plot_sample_image(predictions,x_input = self.dataset['x_test'],y_labels = {'coarse':self.dataset['y_test_coarse'], 'medium':self.dataset['y_test_medium'],'fine':self.dataset['y_test_fine']})
                
            elif 'y_test_coarse' in self.dataset and 'y_test_fine' in self.dataset:

                predictions = self.model.predict([self.dataset['x_test'],self.dataset['y_test_coarse'],self.dataset['y_test_fine']],batch_size = self.batch,verbose=1)
                
                plot_sample_image(predictions,x_input = self.dataset['x_test'],y_labels = {'coarse':self.dataset['y_test_coarse'], 'fine':self.dataset['y_test_fine']})
                
            return predictions
            
def plot_sample_image(predictions, x_input, y_labels : dict, no_sample : int = 10, no_column : int = 10):

            input_shape = x_input.shape[1:] # input shape
            
            plot_row= math.ceil(no_sample/no_column)
            random_number = random.sample(range(0, len(x_input)), no_sample)
            fig, axs = plt.subplots(plot_row,no_column, #### Row and column number for no_sample
                                figsize=(20, 20), facecolor='w', edgecolor='k')
                                
            fig.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()
            
            for i in range(no_sample):
                sample_image = x_input[random_number[i]].reshape(input_shape)
                axs[i].imshow(sample_image)
                
                if len(y_labels) == 2:
                    axs[i].set_title('Input \n Coarse = {0},\nFine = {1}'.format(str(np.argmax(y_labels['coarse'][random_number[i]])), str(np.argmax(y_labels['fine'][random_number[i]]))))
                elif len(y_labels) == 3:
                    axs[i].set_title('Input \n Coarse = {0},\nMedium = {1},\nFine = {2}'.format(str(np.argmax(y_labels['coarse'][random_number[i]])), str(np.argmax(y_labels['medium'][random_number[i]])), str(np.argmax(y_labels['fine'][random_number[i]]))))
            
            fig, axs = plt.subplots(plot_row,no_column, #### Row and column number for no_sample
                    figsize=(20, 20), facecolor='w', edgecolor='k')
                                
            fig.subplots_adjust(hspace = .5, wspace=.001)
            axs = axs.ravel()
            
            for i in range(no_sample):
                
                if len(y_labels) == 2:
                    sample_image = predictions[2][random_number[i]].reshape(input_shape)
                    axs[i].imshow(sample_image)
                    axs[i].set_title('Prediction \n Coarse = {0},\nFine = {1}'.format(str(np.argmax(predictions[0][random_number[i]])), str(np.argmax(predictions[1][random_number[i]]))))
                    
                elif len(y_labels) == 3:
                    sample_image = predictions[3][random_number[i]]
                    axs[i].imshow(sample_image, cmap="binary")

                    axs[i].set_title('Prediction \n Coarse = {0},\nMedium = {1},\nFine = {2}'.format(str(np.argmax(predictions[0][random_number[i]])), str(np.argmax(predictions[1][random_number[i]])), str(np.argmax(predictions[2][random_number[i]]))))
