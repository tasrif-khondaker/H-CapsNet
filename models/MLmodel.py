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
                
def MNIST_HCapsNet(input_shape,
                    no_coarse_class, no_fine_class,
                    PCap_n_dims = 8, SCap_n_dims = 16,
                    n_hidden1 = 512, n_hidden2 = 1024):
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
    # Layer 1 and 2: Two conventional Conv2D layer
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
    # input [batch_size, 6, 6, 512], output [batch_size, 2304, 8]
    reshapec1 = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_c1")(convc11)
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

    reshapef = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    # input [batch_size, 2304, 8], output [batch_size, 10, 16]
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
    model = keras.Model(
        inputs= [x_input, coarse_input, fine_input],
        outputs= [coarse_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet_MNIST')
    
    return model
    
def ENIST_HCapsNet(input_shape,
                    no_fine_class,no_coarse_class,
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
    # Layer 1 and 2: Two conventional Conv2D layer
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
    reshapec1 = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_c1")(convc11)
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

    reshapef = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_fine")(squashcf)


    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    coarse_pred_layer = LengthLayer(name='Coarse_prediction_output_layer')(SecondaryCapsule_coarse)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse_input = keras.Input(shape=(no_coarse_class,), name="coarse_image_label")

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

def FNIST_HCapsNet(input_shape,
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
    # Layer 1 and 2: Two conventional Conv2D layer
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
    reshapec1 = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_c1")(convc11)
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

    reshapec2 = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_c2")(convc23)
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

    reshapef = keras.layers.Reshape((2304, PCap_n_dims), name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse2 = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse2")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse1 = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse1")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    coarse2_pred_layer = LengthLayer(name='Coarse2_prediction_output_layer')(SecondaryCapsule_coarse2)

    # Layer 5.3: Compute the length of each capsule vector
    coarse1_pred_layer = LengthLayer(name='Coarse1_prediction_output_layer')(SecondaryCapsule_coarse1)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse2_input = keras.Input(shape=(no_medium_class,), name="coarse2_image_label")
    coarse1_input = keras.Input(shape=(no_coarse_class,), name="coarse1_image_label")

    # Mask layer
    n_output = np.prod(input_shape)

    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_coarse2, coarse2_input, coarse2_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse1, coarse1_input, coarse1_pred_layer])

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
        outputs= [coarse1_pred_layer, coarse2_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet_Fashion-MNSIT')
    
    return model
    
def CIFAR10_HCapsNet(input_shape,
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
    # Layer 1 and 2: Two conventional Conv2D layer
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

    #Convolution layer for Coarse
    convc11 = keras.layers.Conv2D(filters=256,
                                  kernel_size=6,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((2048, PCap_n_dims), name="reshape_layer_c1")(convc11)
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

    reshapec2 = keras.layers.Reshape((4096, PCap_n_dims), name="reshape_layer_c2")(convc23)
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

    reshapef = keras.layers.Reshape((4096, PCap_n_dims), name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse2 = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse2")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse1 = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse1")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(SecondaryCapsule_fine)

    # Layer 5.2: Compute the length of each capsule vector
    coarse2_pred_layer = LengthLayer(name='Coarse2_prediction_output_layer')(SecondaryCapsule_coarse2)

    # Layer 5.3: Compute the length of each capsule vector
    coarse1_pred_layer = LengthLayer(name='Coarse1_prediction_output_layer')(SecondaryCapsule_coarse1)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse2_input = keras.Input(shape=(no_medium_class,), name="coarse2_image_label")
    coarse1_input = keras.Input(shape=(no_coarse_class,), name="coarse1_image_label")

    # Mask layer
    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_coarse2, coarse2_input, coarse2_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse1, coarse1_input, coarse1_pred_layer])

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
        outputs= [coarse1_pred_layer, coarse2_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet_CIFAR10')

    
    return model
    
def CIFAR100_HCapsNet(input_shape,
                    no_coarse_class, no_medium_class,no_fine_class,
                    PCap_n_dims = 8, SCap_n_dims = 16,
                    n_hidden1 = 512, n_hidden2 = 1024):
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
    # Layer 1 and 2: Two conventional Conv2D layer
    # input [batch_size, 32, 32, 3], output [batch_size, 6, 6, 512]
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

    #Convolution layer for Coarse
    convc11 = keras.layers.Conv2D(filters=256,
                                  kernel_size=6,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)

    # Layer 3: Reshape to 8D primary capsules 
    reshapec1 = keras.layers.Reshape((2048, PCap_n_dims), name="reshape_layer_c1")(convc11)
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

    reshapec2 = keras.layers.Reshape((4096, PCap_n_dims), name="reshape_layer_c2")(convc23)
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

    reshapef = keras.layers.Reshape((4096, PCap_n_dims), name="reshape_layer_f")(convc33)
    squashcf = keras.layers.Lambda(squash, name='squash_layer_f')(reshapef)

    # Layer 4.1: Secondary capsule layer with routing by agreement
    SecondaryCapsule_fine = SecondaryCapsule(n_caps=no_fine_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_fine")(squashcf)

    # Layer 4.2: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse2 = SecondaryCapsule(n_caps=no_medium_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse2")(squashc2)

    # Layer 4.3: Secondary capsule layer with routing by agreement
    SecondaryCapsule_coarse1 = SecondaryCapsule(n_caps=no_coarse_class, n_dims=SCap_n_dims, 
                        name="Digit_Caps_coarse1")(squashc1)

    # Layer 5.1: Compute the length of each capsule vector
    Droupout_layer_fine = keras.layers.Dropout(0.3,name='Fine_prediction_Dropout')(SecondaryCapsule_fine)
    fine_pred_layer = LengthLayer(name='Fine_prediction_output_layer')(Droupout_layer_fine)

    # Layer 5.2: Compute the length of each capsule vector
    coarse2_pred_layer = LengthLayer(name='Coarse2_prediction_output_layer')(SecondaryCapsule_coarse2)

    # Layer 5.3: Compute the length of each capsule vector
    coarse1_pred_layer = LengthLayer(name='Coarse1_prediction_output_layer')(SecondaryCapsule_coarse1)

    fine_input = keras.Input(shape=(no_fine_class,), name="fine_image_label")
    coarse2_input = keras.Input(shape=(no_medium_class,), name="coarse2_image_label")
    coarse1_input = keras.Input(shape=(no_coarse_class,), name="coarse1_image_label")

    # Mask layer

    decoder_input_fine = Mask(name='Mask_input_fine')([SecondaryCapsule_fine, fine_input, fine_pred_layer])
    decoder_input_coarse2 = Mask(name='Mask_input_coarse2')([SecondaryCapsule_coarse2, coarse2_input, coarse2_pred_layer])
    decoder_input_coarse1 = Mask(name='Mask_input_coarse1')([SecondaryCapsule_coarse1, coarse1_input, coarse1_pred_layer])
    
    n_output = np.prod(input_shape)

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
        outputs= [coarse1_pred_layer, coarse2_pred_layer, fine_pred_layer, Output_Layer],
        name='H-CapsNet_CIFAR100') 
    
    return model