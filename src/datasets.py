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



def MNIST(version : str = 'ALL'):
    """
    This is a Manually constructed hierarchical Dataset for MNIST dataset.
    It has 2 hierarchical levels (COARSE and FINE level)
    Coarse classes = 5; Fine Classes = 10.
    :return:
    :X_train:
    :Y_train:
    :X_test:
    :X_test:
    :tree: Digraph
    """
    MNIST = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = MNIST.load_data()
    
    #--- coarse classes ---
    num_coarse = 5
    #--- fine classes ---
    num_fine  = 10
    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    
    #---------------------- make coarse labels --------------------------
    fine_coarse = {0:0, 1:2, 2:1, 3:4, 4:3, 5:4, 6:0, 7:2, 8:1, 9:3}
    y_train_coarse = np.zeros((y_train_fine.shape[0], num_coarse)).astype("float32")
    y_test_coarse = np.zeros((y_test_fine.shape[0], num_coarse)).astype("float32")
    for i in range(y_train_coarse.shape[0]):
        y_train_coarse[i][fine_coarse[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse.shape[0]):
        y_test_coarse[i][fine_coarse[np.argmax(y_test_fine[i])]] = 1.0
        
    tree = DatasetTree([fine_coarse])
    
    if version == 'reduce':
        x_train = x_train[:1000]
        y_train_fine = y_train_fine[:1000]
        y_train_coarse = y_train_coarse[:1000]

        x_test = x_test[:100]
        y_test_fine = y_test_fine[:100]
        y_test_coarse = y_test_coarse[:100]
        print('Using reduced MNIST dataset: Training have 1000 samples and testing have 100 samples')
    else:
        print('MNIST dataset: Training have 60,000 samples and testing have 10,000 samples')
    
    
    # return x_train, y_train_coarse, y_train_fine, x_test, y_test_coarse, y_test_fine, tree
    return {'x_train':x_train, 'y_train_coarse':y_train_coarse,'y_train_fine':y_train_fine, 'x_test':x_test, 'y_test_coarse':y_test_coarse, 'y_test_fine':y_test_fine, 'tree':tree, 'name': 'MNIST'}
    
def E_MNIST(version : str = 'ALL'):
    from emnist import list_datasets, extract_training_samples, extract_test_samples
    print(list_datasets()) ## PRINT contents of the datasets

    x_train, y_train = extract_training_samples('balanced')
    print('Complete loading training samples as: x_train, y_train')
    x_test, y_test = extract_test_samples('balanced')
    print('Complete loading test samples as: x_test, y_test')

    #--- coarse classes ---
    num_coarse = 2
    #--- fine classes ---
    num_fine  = 47

    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    # #---------------- data preprocessiong -------------------
    # x_train = (x_train-np.mean(x_train)) / np.std(x_train)
    # x_test = (x_test-np.mean(x_test)) / np.std(x_test)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    #---------------------- make coarse labels --------------------------
    fine_coarse = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:1,11:1,12:1,13:1,14:1,15:1,16:1,17:1,18:1,19:1,20:1,
                   21:1,22:1,23:1,24:1,25:1,26:1,27:1,28:1,29:1,30:1,31:1,32:1,33:1,34:1,35:1,36:1,37:1,38:1,39:1,
                   40:1,41:1,42:1,43:1,44:1,45:1,46:1}

    y_train_coarse = np.zeros((y_train_fine.shape[0], num_coarse)).astype("float32")
    y_test_coarse = np.zeros((y_test_fine.shape[0], num_coarse)).astype("float32")
    for i in range(y_train_coarse.shape[0]):
        y_train_coarse[i][fine_coarse[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse.shape[0]):
        y_test_coarse[i][fine_coarse[np.argmax(y_test_fine[i])]] = 1.0
        
    tree = DatasetTree([fine_coarse])
    
    if version == 'reduce':
        x_train = x_train[:1000]
        y_train_fine = y_train_fine[:1000]
        y_train_coarse = y_train_coarse[:1000]

        x_test = x_test[:100]
        y_test_fine = y_test_fine[:100]
        y_test_coarse = y_test_coarse[:100]
        print('Using reduced EMNIST dataset: Training have 1000 samples and testing have 100 samples')
    else:
        print('EMNIST dataset: Training have 112,800 samples and testing have 18,800 samples')
    
    # return x_train, y_train_coarse, y_train_fine, x_test, y_test_coarse, y_test_fine, tree
    return {'x_train':x_train, 'y_train_coarse':y_train_coarse,'y_train_fine':y_train_fine, 'x_test':x_test, 'y_test_coarse':y_test_coarse, 'y_test_fine':y_test_fine, 'tree':tree,'name': 'EMNIST'}
    
def F_MNIST(version : str = 'ALL'):
    F_MNIST = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = F_MNIST.load_data()
    #--- coarse 1 classes ---
    num_coarse_1 = 2
    #--- coarse 2 classes ---
    num_coarse_2 = 6
    #--- fine classes ---
    num_fine  = 10
    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    #---------------- data preprocessiong -------------------
    x_train = (x_train-np.mean(x_train)) / np.std(x_train)
    x_test = (x_test-np.mean(x_test)) / np.std(x_test)
    #---------------------- make coarse 2 labels --------------------------
    fine_coarse2 = {0:0, 1:1, 2:0, 3:2, 4:3, 5:5, 6:0, 7:5, 8:4, 9:5}
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32")
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32")
    for i in range(y_train_coarse2.shape[0]):
      y_train_coarse2[i][fine_coarse2[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse2.shape[0]):
      y_test_coarse2[i][fine_coarse2[np.argmax(y_test_fine[i])]] = 1.0
    #---------------------- make coarse 1 labels --------------------------
    coarse2_coarse1 = {0:0, 1:0, 2:0, 3:0, 4:1, 5:1}
    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")
    for i in range(y_train_coarse1.shape[0]):
      y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    for i in range(y_test_coarse1.shape[0]):
      y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0

    tree = DatasetTree([coarse2_coarse1, fine_coarse2])

    if version == 'reduce':
        x_train = x_train[:1000]
        y_train_fine = y_train_fine[:1000]
        y_train_coarse2 = y_train_coarse2[:1000]
        y_train_coarse1 = y_train_coarse1[:1000]

        x_test = x_test[:100]
        y_test_fine = y_test_fine[:100]
        y_test_coarse2 = y_test_coarse2[:100]
        y_test_coarse1 = y_test_coarse1[:100]
        print('Using reduced Fashion-MNIST dataset: Training have 1000 samples and testing have 100 samples')
    else:
        print('Fashion-MNIST dataset: Training have 60,000 samples and testing have 10,000 samples')
        
    return {'x_train':x_train, 'y_train_coarse':y_train_coarse1, 'y_train_medium':y_train_coarse2, 'y_train_fine':y_train_fine, 'x_test':x_test, 'y_test_coarse':y_test_coarse1, 'y_test_medium':y_test_coarse2, 'y_test_fine':y_test_fine, 'tree':tree, 'name': 'FMNIST'}

def CIFAR10(version : str = 'ALL'):
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
    fine_coarse2 = {
      2:3, 3:5, 5:5,
      1:2, 7:6, 4:6,
      0:0, 6:4, 8:1, 9:2
    }
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32")
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32")
    for i in range(y_train_coarse2.shape[0]):
      y_train_coarse2[i][fine_coarse2[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse2.shape[0]):
      y_test_coarse2[i][fine_coarse2[np.argmax(y_test_fine[i])]] = 1.0
      
    #---------------------- make coarse 1 labels --------------------------
    coarse2_coarse1 = {
      0:0, 1:0, 2:0,
      3:1, 4:1, 5:1, 6:1
    }
    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")
    for i in range(y_train_coarse1.shape[0]):
      y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    for i in range(y_test_coarse1.shape[0]):
      y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0

    tree = DatasetTree([coarse2_coarse1, fine_coarse2])

    if version == 'reduce':
        x_train = x_train[:1000]
        y_train_fine = y_train_fine[:1000]
        y_train_coarse2 = y_train_coarse2[:1000]
        y_train_coarse1 = y_train_coarse1[:1000]

        x_test = x_test[:100]
        y_test_fine = y_test_fine[:100]
        y_test_coarse2 = y_test_coarse2[:100]
        y_test_coarse1 = y_test_coarse1[:100]
        print('Using reduced CIFAR-10 dataset: Training have 1000 samples and testing have 100 samples')
    else:
        print('CIFAR-10 dataset: Training have 50,000 samples and testing have 10,000 samples')
        
    return {'x_train':x_train, 'y_train_coarse':y_train_coarse1, 'y_train_medium':y_train_coarse2, 'y_train_fine':y_train_fine, 'x_test':x_test, 'y_test_coarse':y_test_coarse1, 'y_test_medium':y_test_coarse2, 'y_test_fine':y_test_fine, 'tree':tree, 'name': 'CIFAR-10'}
    
def CIFAR100(version : str = 'ALL'):
    CIFAR100 = keras.datasets.cifar100

    (x_train, y_train), (x_test, y_test) = CIFAR100.load_data(label_mode='fine')
    #--- coarse 1 classes ---
    num_coarse_1 = 8
    #--- coarse 2 classes ---
    num_coarse_2 = 20
    #--- fine classes ---
    num_fine  = 100

    #-------------------- data loading ----------------------
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train_fine = keras.utils.to_categorical(y_train, num_fine)
    y_test_fine = keras.utils.to_categorical(y_test, num_fine)

    #---------------- data preprocessiong -------------------
    x_train = (x_train-np.mean(x_train)) / np.std(x_train)
    x_test = (x_test-np.mean(x_test)) / np.std(x_test)

    fine_coarse2 = {
    0:4,1:1,2:14,3:8,4:0,5:6,6:7,7:7,8:18,9:3,
    10:3,11:14,12:9,13:18,14:7,15:11,16:3,17:9,18:7,19:11,
    20:6,21:11,22:5,23:10,24:7,25:6,26:13,27:15,28:3,29:15,
    30:0,31:11,32:1,33:10,34:12,35:14,36:16,37:9,38:11,39:5,
    40:5,41:19,42:8,43:8,44:15,45:13,46:14,47:17,48:18,49:10,
    50:16,51:4,52:17,53:4,54:2,55:0,56:17,57:4,58:18,59:17,
    60:10,61:3,62:2,63:12,64:12,65:16,66:12,67:1,68:9,69:19,
    70:2,71:10,72:0,73:1,74:16,75:12,76:9,77:13,78:15,79:13,
    80:16,81:19,82:2,83:4,84:6,85:19,86:5,87:5,88:8,89:19,
    90:18,91:1,92:2,93:15,94:6,95:0,96:17,97:8,98:14,99:13
    }
    y_train_coarse2 = np.zeros((y_train_fine.shape[0], num_coarse_2)).astype("float32")
    y_test_coarse2 = np.zeros((y_test_fine.shape[0], num_coarse_2)).astype("float32")
    for i in range(y_train_coarse2.shape[0]):
      y_train_coarse2[i][fine_coarse2[np.argmax(y_train_fine[i])]] = 1.0
    for i in range(y_test_coarse2.shape[0]):
      y_test_coarse2[i][fine_coarse2[np.argmax(y_test_fine[i])]] = 1.0
      
    #---------------------- make coarse 1 labels --------------------------
    coarse2_coarse1 = {
      0:0, 1:0, 2:1, 3:2, 
      4:1, 5:2, 6:2, 7:3, 
      8:4, 9:5, 10:5, 11:4, 
      12:4, 13:3, 14:6, 15:4, 
      16:4, 17:1, 18:7, 19:7
    }
    y_train_coarse1 = np.zeros((y_train_coarse2.shape[0], num_coarse_1)).astype("float32")
    y_test_coarse1 = np.zeros((y_test_coarse2.shape[0], num_coarse_1)).astype("float32")
    for i in range(y_train_coarse1.shape[0]):
      y_train_coarse1[i][coarse2_coarse1[np.argmax(y_train_coarse2[i])]] = 1.0
    for i in range(y_test_coarse1.shape[0]):
      y_test_coarse1[i][coarse2_coarse1[np.argmax(y_test_coarse2[i])]] = 1.0
      
    tree = DatasetTree([coarse2_coarse1, fine_coarse2])

    if version == 'reduce':
        x_train = x_train[:1000]
        y_train_fine = y_train_fine[:1000]
        y_train_coarse2 = y_train_coarse2[:1000]
        y_train_coarse1 = y_train_coarse1[:1000]

        x_test = x_test[:100]
        y_test_fine = y_test_fine[:100]
        y_test_coarse2 = y_test_coarse2[:100]
        y_test_coarse1 = y_test_coarse1[:100]
        print('Using reduced CIFAR-100 dataset: Training have 1000 samples and testing have 100 samples')
    else:
        print('CIFAR-100 dataset: Training have 50,000 samples and testing have 10,000 samples')
        
    return {'x_train':x_train, 'y_train_coarse':y_train_coarse1, 'y_train_medium':y_train_coarse2, 'y_train_fine':y_train_fine, 'x_test':x_test, 'y_test_coarse':y_test_coarse1, 'y_test_medium':y_test_coarse2, 'y_test_fine':y_test_fine, 'tree':tree, 'name': 'CIFAR-100'}
    
def plot_sample_image(x_input, y_labels : dict, no_sample : int = 10, no_column : int = 10):

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
            axs[i].set_title('Coarse = {0},\nFine = {1}'.format(str(np.argmax(y_labels['coarse'][random_number[i]])), str(np.argmax(y_labels['fine'][random_number[i]]))))
        elif len(y_labels) == 3:
            axs[i].set_title('Coarse = {0},\nMedium = {1},\nFine = {2}'.format(str(np.argmax(y_labels['coarse'][random_number[i]])), str(np.argmax(y_labels['medium'][random_number[i]])), str(np.argmax(y_labels['fine'][random_number[i]]))))
        
print("DONE loading Datasets")

class Marine_Dataset:
    
    def __init__(self, name, dataset_path, train_labels_path, test_labels_path,output_level, 
                 image_size=(64, 64), batch_size=32, 
                 data_normalizing:str='normalize', class_encoding:str = 'Label_Encoder',
                 data_augmantation:str = 'mixup', mixup_alpha:float = 0.2):
        
        self.name = name
        self.image_size_ = image_size
        self.image_size = (image_size[0], image_size[1], 3)
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.output_level = output_level
        
        # Training set
        train_labels_df = pd.read_csv(train_labels_path, sep=",", header=0)
        train_labels_df = train_labels_df.sample(frac=1).reset_index(drop=True)
        self.train_labels_df = train_labels_df
        # Splitting into val and test sets
        test_labels_df = pd.read_csv(test_labels_path, sep=",", header=0)
        test_labels_df = test_labels_df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.test_labels_df = test_labels_df
        
        self.class_encoding = class_encoding
        self.data_normalizing = data_normalizing
        self.data_augmantation = data_augmantation
        self.mixup_alpha = mixup_alpha
        
        # Number of classes
        self.num_classes_l0 = len(set(train_labels_df['class_level_0']))
        self.num_classes_l1 = len(set(train_labels_df['class_level_1']))
        self.num_classes_l2 = len(set(train_labels_df['class_level_2']))
        self.num_classes_l3 = len(set(train_labels_df['class_level_3']))
        self.num_classes_l4 = len(set(train_labels_df['class_level_4']))
        
        train_labels_df,val_labels_df = train_test_split(train_labels_df, test_size=0.12,random_state=42, stratify=train_labels_df['class_level_4'])
        
        self.train_labels_df = train_labels_df
        
        self.val_labels_df = val_labels_df
        
        #Train set pipeline
        self.train_dataset = self.get_pipeline(train_labels_df,output_level,data_aug = self.data_augmantation, 
                                               alpha_val= self.mixup_alpha)
        # Test set pipeline
        self.test_dataset = self.get_pipeline(test_labels_df,output_level, data_aug = None, alpha_val= None)
        # Validation set pipeline
        self.val_dataset = self.get_pipeline(val_labels_df,output_level, data_aug = None, alpha_val= None)

        
        # Encoding the taxonomy
        m0 = [[0 for x in range(self.num_classes_l1)] for y in range(self.num_classes_l0)]
        
        for (t, c) in zip(list(train_labels_df['class_level_0']), list(train_labels_df['class_level_1'])):
            m0[t][c] = 1
        
        m1 = [[0 for x in range(self.num_classes_l2)] for y in range(self.num_classes_l1)]
        for (t, c) in zip(list(train_labels_df['class_level_1']), list(train_labels_df['class_level_2'])):
            m1[t][c] = 1
            
        m2 = [[0 for x in range(self.num_classes_l3)] for y in range(self.num_classes_l2)]
        for (t, c) in zip(list(train_labels_df['class_level_2']), list(train_labels_df['class_level_3'])):
            m2[t][c] = 1
            
        m3 = [[0 for x in range(self.num_classes_l4)] for y in range(self.num_classes_l3)]
        for (t, c) in zip(list(train_labels_df['class_level_3']), list(train_labels_df['class_level_4'])):
            m3[t][c] = 1
       
       
        # Build the labels
        self.labels = [] ## define empty labels
        
        if self.output_level == 'only_level_3':
            self.num_classes = [self.num_classes_l2]
            self.taxonomy = [m3]
            
            # Build the labels
            labels = ['' for x in range(self.num_classes_l2)]
            for (l, c) in zip(list(train_labels_df['label_level_2']), list(train_labels_df['class_level_2'])):
                labels[c] = l
            self.labels.append(labels)
        
        if self.output_level == 'level_depth_3':
            self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2]
            self.taxonomy = [m0, m1]
            
            # Build the labels
            labels = ['' for x in range(self.num_classes_l0)]
            for (l, c) in zip(list(train_labels_df['label_level_0']), list(train_labels_df['class_level_0'])):
                labels[c] = l
            self.labels.append(labels)

            labels = ['' for x in range(self.num_classes_l1)]
            for (l, c) in zip(list(train_labels_df['label_level_1']), list(train_labels_df['class_level_1'])):
                labels[c] = l
            self.labels.append(labels)

            labels = ['' for x in range(self.num_classes_l2)]
            for (l, c) in zip(list(train_labels_df['label_level_2']), list(train_labels_df['class_level_2'])):
                labels[c] = l
            self.labels.append(labels)
        
        if self.output_level == 'level_depth_5':
            self.num_classes = [self.num_classes_l0, self.num_classes_l1, self.num_classes_l2,
                                self.num_classes_l3,self.num_classes_l4]     
            self.taxonomy = [m0, m1,m2,m3]
            
            # Build the labels
        
            labels = ['' for x in range(self.num_classes_l0)]
            for (l, c) in zip(list(train_labels_df['label_level_0']), list(train_labels_df['class_level_0'])):
                labels[c] = l
            self.labels.append(labels)

            labels = ['' for x in range(self.num_classes_l1)]
            for (l, c) in zip(list(train_labels_df['label_level_1']), list(train_labels_df['class_level_1'])):
                labels[c] = l
            self.labels.append(labels)

            labels = ['' for x in range(self.num_classes_l2)]
            for (l, c) in zip(list(train_labels_df['label_level_2']), list(train_labels_df['class_level_2'])):
                labels[c] = l
            self.labels.append(labels)

            labels = ['' for x in range(self.num_classes_l3)]
            for (l, c) in zip(list(train_labels_df['label_level_3']), list(train_labels_df['class_level_3'])):
                labels[c] = l
            self.labels.append(labels)

            labels = ['' for x in range(self.num_classes_l4)]
            for (l, c) in zip(list(train_labels_df['label_level_4']), list(train_labels_df['class_level_4'])):
                labels[c] = l
            self.labels.append(labels)
        
        
    def encode_single_sample(self, img_path, class_level_0, class_level_1, class_level_2, 
                             class_level_3, class_level_4,fname,output_level):

        # 1. Read image
        img = tf.io.read_file(img_path)
        # 2. Decode and convert to grayscale
        img = tf.io.decode_image(img, expand_animations=False)
        # 3. Convert to float32 in [0, 1] range
        img = tf.image.convert_image_dtype(img, tf.float32)
        # 4. Resize to the desired size
        img = tf.image.resize(img, self.image_size_)
        
        # standardize a dataset with a certain mean and standard deviation value
        if self.data_normalizing == 'StandardScaler':
            img = normalize(img)
        
        #One hot encoding
        if self.class_encoding == 'One_Hot_Encoder':
            
            class_level_0 = tf.one_hot(indices=class_level_0, depth = int(self.num_classes_l0))
            class_level_1 = tf.one_hot(indices=class_level_1, depth = self.num_classes_l1)
            class_level_2 = tf.one_hot(indices=class_level_2, depth = self.num_classes_l2)
            class_level_3 = tf.one_hot(indices=class_level_3, depth = self.num_classes_l3)
            class_level_4 = tf.one_hot(indices=class_level_4, depth = self.num_classes_l4)
        
        if self.output_level == 'only_level_3':
            return img, class_level_2
        
        if self.output_level == 'level_depth_3':
            return img, (class_level_0, class_level_1, class_level_2)
        
        if self.output_level == 'level_depth_5':
            return img, (class_level_0, class_level_1, class_level_2, class_level_3, class_level_4)


    def get_pipeline(self, dataframe, output_level, data_aug:str = 'mixup', alpha_val:float = 0.2):
        
        self.output_level = output_level
        
        if data_aug == 'mixup':
            ### For mixup, when creating a dataset pipeline it should be 1st shuffle then batch. 
            
            ## dataset - 1
            ds_one = tf.data.Dataset.from_tensor_slices(([self.dataset_path +'/marine_images/' + x for x in dataframe['fname']],
                                                          list(dataframe['class_level_0']),
                                                          list(dataframe['class_level_1']),
                                                          list(dataframe['class_level_2']),
                                                          list(dataframe['class_level_3']),
                                                          list(dataframe['class_level_4']),
                                                          list(dataframe['fname']),
                                                          [output_level for x in dataframe['fname']]))
            ds_one = (ds_one.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                            .shuffle(self.batch_size * 100).padded_batch(self.batch_size)
                            .prefetch(buffer_size=tf.data.AUTOTUNE))
            ## dataset - 2
            ds_two = tf.data.Dataset.from_tensor_slices(([self.dataset_path +'/marine_images/' + x for x in dataframe['fname']],
                                                          list(dataframe['class_level_0']),
                                                          list(dataframe['class_level_1']),
                                                          list(dataframe['class_level_2']),
                                                          list(dataframe['class_level_3']),
                                                          list(dataframe['class_level_4']),
                                                          list(dataframe['fname']),
                                                          [output_level for x in dataframe['fname']]))
            ds_two = (ds_two.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                            .shuffle(self.batch_size * 100).padded_batch(self.batch_size)
                            .prefetch(buffer_size=tf.data.AUTOTUNE))
            
            ds = tf.data.Dataset.zip((ds_one, ds_two))
            
            dataset = ds.map(
                lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=alpha_val), num_parallel_calls=tf.data.AUTOTUNE
            )
            
        else:
            dataset = tf.data.Dataset.from_tensor_slices(([self.dataset_path +'/marine_images/' + x for x in dataframe['fname']],
                                                          list(dataframe['class_level_0']),
                                                          list(dataframe['class_level_1']),
                                                          list(dataframe['class_level_2']),
                                                          list(dataframe['class_level_3']),
                                                          list(dataframe['class_level_4']),
                                                          list(dataframe['fname']),
                                                          [output_level for x in dataframe['fname']]))


            dataset = (dataset.map(self.encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
                       .padded_batch(self.batch_size)
                       .prefetch(buffer_size=tf.data.AUTOTUNE)
                      )
        
        return dataset

    def get_tree(self):
        return get_tree(self.taxonomy, self.labels)
        
def get_Marine_dataset(output_level, dataset_path:str, image_size=(64, 64), batch_size=32, subtype='Tropical',
                       data_normalizing ='normalize', class_encoding = 'Label_Encoder', 
                       data_augmantation = 'mixup', mixup_alpha = 0.2):
    # Get images
    
    dataset_path = dataset_path
    
    if subtype == 'Tropical':
        
        train_labels_path = dataset_path+'\\train_labels_trop.csv'
        test_labels_path = dataset_path+'\\test_labels_trop.csv'
        
    elif subtype == 'Temperate':
        
        train_labels_path = dataset_path+'\\train_labels_temp.csv'
        test_labels_path = dataset_path+'\\test_labels_temp.csv'
        
    else:
        
        train_labels_path = dataset_path+'\\train_labels_comb.csv'
        test_labels_path = dataset_path+'\\test_labels_comb.csv'
        
    
    dataset_name = 'Marine_dataset_'+subtype


    return Marine_Dataset(dataset_name, dataset_path, train_labels_path, test_labels_path,output_level, image_size, 
                          batch_size, data_normalizing, class_encoding, data_augmantation, mixup_alpha)