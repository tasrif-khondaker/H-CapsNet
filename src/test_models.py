                
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
    conv2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool')(conv2)

    #Convolution layer for coarse

    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)
    convc11 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_1')(convc11)

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
    convc33 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_3')(convc33)

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
    conv2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool')(conv2)

    #Convolution layer for coarse
    convc11 = keras.layers.Conv2D(filters=512,
                                  kernel_size=7,
                                  padding="valid",
                                  strides=3,
                                  activation=tf.nn.relu,
                                  name='convc11')(conv2)
    convc11 = keras.layers.BatchNormalization()(convc11)
    convc11 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_1')(convc11)

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
    convc33 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_3')(convc33)

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
    conv2 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool')(conv2)

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
    convc23 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_2')(convc23)

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
    convc33 = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='maxpool_3')(convc33)

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
