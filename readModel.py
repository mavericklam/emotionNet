import os

import numpy as np

import h5py

import tensorflow as tf


import matplotlib.pyplot as plt

import scipy.misc

import pandas as pd

print (tf.__version__)

PARAMS_PATH = ''
PARAMS_FILE = PARAMS_PATH + 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'
data_h5 = h5py.File(PARAMS_FILE, 'r')

variables = [ key for key in data_h5.keys() if len(data_h5[key])>0 ]



def conv_layer(input_layer, data, layer_name, strides=[1,1,1,1], padding='VALID', trainable = False):
    with tf.variable_scope(layer_name):
        W_val = np.array(data[layer_name][layer_name+'_W:0']).astype(np.float32)
        b_val = np.array(data[layer_name][layer_name+'_b:0']).astype(np.float32)
        if trainable:
            W = tf.get_variable(layer_name+'_W', shape=W_val.shape,
                                initializer=tf.constant_initializer(W_val), dtype=tf.float32)
            b = tf.get_variable(layer_name+'_b', shape=b_val.shape,
                                initializer=tf.constant_initializer(b_val), dtype=tf.float32)
        else:
            W = tf.constant( W_val )
            b = tf.constant( np.reshape(b_val, (b_val.shape[0])) )

        X = tf.nn.conv2d(input_layer, filter=W, strides=strides, padding=padding, name=layer_name)
        X = tf.nn.bias_add(X, b)
        return X



def batch_norm_layer(input_layer, data, layer_name, trainable = False):
    with tf.variable_scope(layer_name):
        mean_val = np.array(data[layer_name][layer_name+'_running_mean:0']).astype(np.float32)
        std_val = np.array(data[layer_name][layer_name+'_running_std:0']).astype(np.float32)
        beta_val = np.array(data[layer_name][layer_name+'_beta:0']).astype(np.float32)
        gamma_val = np.array(data[layer_name][layer_name+'_gamma:0']).astype(np.float32)

        if trainable:
            mean = tf.get_variable(layer_name+'_running_mean', shape=mean_val.shape, initializer=tf.constant_initializer(mean_val), dtype=tf.float32)
            std = tf.get_variable(layer_name+'_running_std', shape=std_val.shape, initializer=tf.constant_initializer(std_val), dtype=tf.float32)
            beta = tf.get_variable(layer_name+'_beta', shape=beta_val.shape, initializer=tf.constant_initializer(beta_val), dtype=tf.float32)
            gamma = tf.get_variable(layer_name+'_gamma', shape=gamma_val.shape, initializer=tf.constant_initializer(gamma_val), dtype=tf.float32)
        else:
            mean = tf.constant(mean_val)
            std = tf.constant(std_val)
            beta = tf.constant(beta_val)
            gamma = tf.constant(gamma_val)
            # As variables.
        X = tf.nn.batch_normalization( input_layer, mean=mean, variance=std,
                                       offset=beta, scale=gamma, variance_epsilon=1e-12, name='batch-norm')
        return X

def identity_block(input_layer, stage, data,trainable = False):

    with tf.variable_scope('identity_block'):
        x = conv_layer(input_layer, data=data, layer_name='res'+stage+'_branch2a',trainable=trainable)
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2a',trainable=trainable)
        x = tf.nn.relu(x)

        x = conv_layer(x, data=data, layer_name='res'+stage+'_branch2b', padding='SAME',trainable=trainable)
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2b',trainable=trainable)
        x = tf.nn.relu(x)

        x = conv_layer(x, data=data, layer_name='res'+stage+'_branch2c',trainable=trainable)
        x = batch_norm_layer(x, data=data, layer_name='bn'+stage+'_branch2c',trainable=trainable)

        x = tf.add(x, input_layer)
        x = tf.nn.relu(x)
    return x

def conv_block(input_layer, stage, data, strides=[1, 2, 2, 1],trainable = False):
    with tf.variable_scope('conv_block'):
        x = conv_layer(input_layer = input_layer, data = data, layer_name = 'res'+stage+'_branch2a',strides=strides,trainable=trainable)
        x = batch_norm_layer(input_layer = x, data = data, layer_name= 'bn'+stage+'_branch2a',trainable=trainable)
        x = tf.nn.relu(x)
        x = conv_layer(input_layer = x, data = data, layer_name = 'res'+stage+'_branch2b',padding='SAME',trainable=trainable)
        x = batch_norm_layer(input_layer = x, data = data, layer_name= 'bn'+stage+'_branch2b',trainable=trainable)
        x = tf.nn.relu(x)
        x = conv_layer(input_layer = x, data = data, layer_name = 'res'+stage+'_branch2c',trainable=trainable)
        x = batch_norm_layer(input_layer = x, data = data, layer_name= 'bn'+stage+'_branch2c',trainable=trainable)
        shortcut = conv_layer(input_layer = input_layer, data = data, layer_name = 'res'+stage+'_branch1',strides=strides,trainable=trainable)
        shortcut = batch_norm_layer(input_layer = shortcut, data = data, layer_name = 'bn'+stage+'_branch1',trainable=trainable)
        x = tf.add(shortcut,x)
        x = tf.nn.relu(x)
    return x

# The function to create the fully connected layer that is needed at the end.
def dense_layer(input_layer, data, layer_name,trainable = False):
    with tf.variable_scope(layer_name):
        W_val = tf.constant( data[layer_name][layer_name+'_W:0'] )
        b_val = data[layer_name][layer_name+'_b:0']
        if trainable:
            W = tf.get_variable(layer_name+'_W', shape=W_val.shape, initializer=tf.constant_initializer(W_val))
            b = tf.get_variable(layer_name+'_bias', shape=b_val.shape, initializer=tf.constant_initializer(b_val))
        else:
            W = tf.constant( data[layer_name][layer_name+'_W:0'] )
            b = data[layer_name][layer_name+'_b:0']
        b = tf.constant( np.reshape(b, (b.shape[0])) )
        x = tf.matmul(input_layer, W)
        x = tf.nn.bias_add(x, b)
        return x

tf.reset_default_graph()

RESNET_HEIGHT = 224
RESNET_WIDTH = 224

image_input = tf.placeholder(dtype=tf.float32, shape=[None, RESNET_HEIGHT, RESNET_WIDTH, 3], name='input')
image = tf.pad(image_input, [[0,0],[3,3],[3,3],[0,0]], "CONSTANT", name='zeropadding-3')

with tf.variable_scope('stage1'):
    res = conv_layer(image, data_h5, 'conv1', strides=[1, 2, 2, 1], trainable = False)
    res = batch_norm_layer(res, data_h5, 'bn_conv1', trainable = False)
    res = tf.nn.relu(res)
    res = tf.nn.max_pool(res, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_conv1')
    print ('Stage 1', res.get_shape())

with tf.variable_scope('stage2'):
    res = conv_block(input_layer=res, stage='2a', data=data_h5, strides=[1, 1, 1, 1], trainable = False)
    res = identity_block(input_layer=res, stage='2b', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='2c', data=data_h5, trainable = False)
    print ('Stage 2', res.get_shape())

with tf.variable_scope('stage3'):
    res = conv_block(input_layer=res, stage='3a', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='3b', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='3c', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='3d', data=data_h5, trainable = False)
    print ('Stage 3', res.get_shape())

with tf.variable_scope('stage4'):
    res = conv_block(input_layer=res, stage='4a', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='4b', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='4c', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='4d', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='4e', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='4f', data=data_h5, trainable = False)
    print ('Stage 4', res.get_shape())

with tf.variable_scope('stage5'):
    res = conv_block(input_layer=res, stage='5a', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='5b', data=data_h5, trainable = False)
    res = identity_block(input_layer=res, stage='5c', data=data_h5, trainable = True)
    print ('Stage 5', res.get_shape())


with tf.variable_scope('stage-final'):
    res = tf.nn.avg_pool(res, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool_conv1')
    print ('Pool 5', res.get_shape())
    # Add the dense layer.
    res = tf.reshape(res, (-1, res.get_shape()[3].value))
    res = dense_layer(input_layer=res, layer_name='fc1000', data=data_h5, trainable = False)
    res = tf.nn.softmax(res)
    print ('Output probabilities', res.get_shape())


def load_image(path):

    image = scipy.misc.imread(path)

    # Crop to square.
    min_dim = np.min(image.shape[:2])
    height_dim0 = (image.shape[0]-min_dim) // 2
    height_dim1 = image.shape[0] + ((image.shape[0]-min_dim) // 2)
    width_dim0 = (image.shape[1]-min_dim) // 2
    width_dim1 = min_dim + ((image.shape[1]-min_dim) // 2)
    image_cropped = image[height_dim0:height_dim1, width_dim0:width_dim1, :]

    # Resize to ResNet input dimensions.
    image_resized = scipy.misc.imresize(image_cropped, [RESNET_HEIGHT, RESNET_WIDTH])

    return image_resized

with tf.Session(graph=tf.get_default_graph()) as sess:
    # train your model
    saver = tf.train.Saver()
    saver.save(sess, 'model/ResNet50.ckpt')

sess = tf.InteractiveSession()






# Get the imagenet labels
with open('./labels/imagenet-labels-1001.txt') as file: # First row is a dummy.
    labels = file.readlines()
labels = [ x.strip() for x in labels ]
df = pd.DataFrame({ 'labels':labels[1:], 'probabilities': np.zeros((1000,)) })

# Set the path to the images to load.
images = ['./images/abc.jpg']


plt.figure(figsize=(16,6))

for i, image_path in enumerate(images):

    image = load_image(image_path)

    probs = sess.run(
        tf.get_default_graph().get_tensor_by_name("stage-final/Softmax:0"),
        feed_dict={ image_input: image.reshape((1, RESNET_HEIGHT, RESNET_WIDTH, 3)) } )
    plt.subplot(1, len(images), i+1)
    plt.imshow(image)
    ixs = probs[0].argsort()[-3:][::-1]
    title = ''
    for j,ix in enumerate(ixs):
        if j > 0:
            title += '\n'
        title += df['labels'].loc[ix] + ' ({:3.2f}%)'.format(probs[0,ix]*100)
    plt.title(title)
    plt.show()
    print(title)
