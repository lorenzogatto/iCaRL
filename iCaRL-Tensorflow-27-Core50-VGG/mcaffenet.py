import tensorflow as tf
import numpy as np
import cPickle
import network
import sys

def create_model(inp, phase, num_outputs=1000, alpha=0.0):
    trainable = False
    if phase == 'train':
        trainable = True
    # print(phase)
    global net
    net = network.Network(inputs=inp, trainable=trainable)
    layer = net.feed('data') \
        .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')\
        .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')\
        .lrn(2, 2e-05, 0.75, name='norm1')\
        .conv(5, 5, 256, 1, 1, group=2, name='conv2')\
        .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')\
        .lrn(2, 2e-05, 0.75, name='norm2')\
        .conv(3, 3, 384, 1, 1, name='conv3')\
        .conv(3, 3, 384, 1, 1, group=2, name='conv4')\
        .conv(3, 3, 256, 1, 1, group=2, name='conv5')\
        .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')\
        .fc(2048, name='mid_fc6')\
        .dropout(0.5, name='drop6')\
        .fc(2048, name='mid_fc7')\
        .dropout(0.5, name='drop7')\
        .fc(50, relu=False, name='mid_fc8')\
        .get_output()
    return layer

def initialize_imagenet(sess):
     with tf.variable_scope('ResNet18'):
         mcaffenet_npy = 'alexnet.npy'
         net.load(mcaffenet_npy, sess, skip_op=['fc6', 'fc7', 'fc8'])


def get_weight_initializer(params):
     initializer = []

     scope = tf.get_variable_scope()
     scope.reuse_variables()
     for layer, value in params.items():
          op = tf.get_variable('%s' % layer).assign(value)
          initializer.append(op)
     return initializer


def save_model(name, scope, sess):
     variables = tf.get_collection(tf.GraphKeys.WEIGHTS, scope=scope)
     d = [(v.name.split(':')[0], sess.run(v)) for v in variables]

     cPickle.dump(d, open(name, 'w'), protocol=2)
