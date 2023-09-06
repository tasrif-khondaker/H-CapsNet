import tensorflow as tf
from tensorflow import keras
from treelib import Tree
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from emnist import list_datasets, extract_training_samples, extract_test_samples
import math
import random
import matplotlib
import matplotlib.pyplot as plt


def DatasetTree(level_maps):
    """
    provide levels maps as a list of class level maps! Prioritising from coarse to fine levels.
    Works only for hierarchy with 2 and 3 levels.
    """
    levels = len(level_maps)
    
    tree = Tree()
    tree.create_node("Root", "root")  # root node

        
    tree = Tree()
    tree.create_node("Root", "root")  # root node
    
    if levels == 1:
        for i in range(len(set(level_maps[0].values()))):
            tree.create_node('Coarse_'+str(i), 'L0_'+ str(i), parent="root")
            for j in range(len(level_maps[0])):
                if level_maps[0][j] == i :
                    tree.create_node('Fine_'+str(j), 'L1_'+str(j), 'L0_'+ str(i))
                    
    elif levels == 2:
        for i in range(len(set(level_maps[0].values()))):
            tree.create_node('Coarse_'+str(i), 'L0_'+ str(i), parent="root")
            for j in range(len(level_maps[0])):
                if level_maps[0][j] == i :
                    tree.create_node('Medium'+str(j), 'L1_'+str(j), 'L0_'+ str(i))
                    for k in range(len(level_maps[1])):
                        if level_maps[1][k] == j :
                            tree.create_node('Fine_'+str(k), 'L2_'+str(k), 'L1_'+ str(j))
                
    return tree
    
def get_tree(taxonomy, labels):
    """
    This method draws the taxonomy using the graphviz library.
    :return:
    :rtype: Digraph
     """
    tree = Tree()
    tree.create_node("Root", "root")  # root node
    
    if len(taxonomy) > 1:
        for i in range(len(taxonomy[0])):
            tree.create_node(labels[0][i] + ' -> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

        for l in range(len(taxonomy)):
            for i in range(len(taxonomy[l])):
                for j in range(len(taxonomy[l][i])):
                    if taxonomy[l][i][j] == 1:
                        tree.create_node(labels[l + 1][j] + ' -> (L' + str(l + 1) + '_' + str(j) + ')',
                                         'L' + str(l + 1) + '_' + str(j),
                                         parent='L' + str(l) + '_' + str(i))
    if len(taxonomy) == 1:
        for i in range(len(labels[0])):
            tree.create_node(labels[0][i] + ' -> (L0_' + str(i) + ')', 'L0_' + str(i), parent="root")

    return tree

def normalize(data):
    data = tf.cast(data, dtype=tf.float32) # Convert to float 32
    mean = tf.math.reduce_mean(data)
    std = tf.math.reduce_std(data)
    data = tf.subtract(data, mean)
    data = tf.divide(data, std)
    return data

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = [i for i in range(len(labels_one))]
    for i in range(len(labels_one)):
        labels[i] = labels_one[i] * y_l + labels_two[i] * (1 - y_l)
        
    return (images, tuple(labels)) 

def CIFAR10_DAG():
    '''
    Based on the CIFAR10 dataset, this function generates a DAG dataset.
    this is similar to CIFAR10() but the fine_coarse2 dictionary is changed to a DAG.
    As a result, the y_train_coarse2 y_test_coarse2 are different. In this case, the one-hot encoding have multiple 1s for sample 2 and 6.
    '''
    CIFAR10 = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = CIFAR10.load_data()
    #--- coarse 1 classes ---
    num_coarse_1 = 2
    #--- coarse 2 classes ---
    num_coarse_2 = 7
    #--- fine classes ---
    num_fine  = 10

    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    #---------------- data preprocessiong -------------------
    x_train = (x_train-np.mean(x_train)) / np.std(x_train)
    x_test = (x_test-np.mean(x_test)) / np.std(x_test)

    #---------------------- make coarse 2 labels --------------------------

    fine_coarse2 = { # DAG
        0:[0], 
        1:[2], 
        2:[3,5], 
        3:[5], 
        4:[6],
        5:[5],
        6:[4], 
        7:[6, 2], 
        8:[1], 
        9:[2]
    }
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32")
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32")

    for i, fine_label in enumerate(y_train_fine):
        medium_labels = fine_coarse2[np.argmax(fine_label)]
        for medium_label in range(num_coarse_2):
            if medium_label in medium_labels:
                y_train_coarse2[i, medium_label] = 1.0


    for i, fine_label in enumerate(y_test_fine):
        medium_labels = fine_coarse2[np.argmax(fine_label)]
        for medium_label in medium_labels:
            y_test_coarse2[i, medium_label] = 1.0

    #---------------------- make coarse 1 labels --------------------------
    coarse2_coarse1 = { # DAG
        0:[0], 
        1:[0], 
        2:[0],
        3:[1], 
        4:[1], 
        5:[1], 
        6:[1]
    }

    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")

    for i, medium_label in enumerate(y_train_coarse2):
        coarse_labels = coarse2_coarse1[np.argmax(medium_label)]
        for coarse_label in coarse_labels:
            y_train_coarse1[i, coarse_label] = 1.0

    for i, medium_label in enumerate(y_test_coarse2):
        coarse_labels = coarse2_coarse1[np.argmax(medium_label)]
        for coarse_label in coarse_labels:
            y_test_coarse1[i, coarse_label] = 1.0

    # for i in range(y_train_coarse1.shape[0]):
    #     y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    # for i in range(y_test_coarse1.shape[0]):
    #     y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0

    # tree = DatasetTree([coarse2_coarse1, fine_coarse2])
    tree = None

    print('CIFAR-10 DAG dataset: Training have 50,000 samples and testing have 10,000 samples')
    return {'x_train':x_train, 'y_train_coarse':y_train_coarse1, 'y_train_medium':y_train_coarse2, 'y_train_fine':y_train_fine, 'x_test':x_test, 'y_test_coarse':y_test_coarse1, 'y_test_medium':y_test_coarse2, 'y_test_fine':y_test_fine, 'tree':tree, 'name': 'CIFAR-10_DAG'}


print("DONE loading Datasets")
