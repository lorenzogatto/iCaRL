import tensorflow as tf
import numpy as np
import cPickle
import network
import sys
'''
Implementation of MidVGGM as explained in Core50 paper
'''
def create_model(inp, phase, num_outputs=1000, alpha=0.0):
    trainable = False
    if phase == 'train':
        trainable = True
    #print(phase)
    global net
    net = network.Network(inputs=inp, trainable=trainable)
    layer = net.feed('data')\
        .conv(7, 7, 96, 2, 2, padding='VALID', name='conv1')\
        .lrn(2, 0.00010000000475, 0.75, name='norm1')\
        .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')\
        .pad(1)\
        .conv(5, 5, 256, 2, 2, padding='VALID', name='conv2')\
        .lrn(2, 0.00010000000475, 0.75, name='norm2')\
        .max_pool(3, 3, 2, 2, name='pool2')\
        .conv(3, 3, 512, 1, 1, name='conv3')\
        .conv(3, 3, 512, 1, 1, name='conv4')\
        .conv(3, 3, 512, 1, 1, name='conv5')\
        .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')\
        .fc(2048, name='mid_fc6')\
        .dropout(0.5)\
        .fc(2048, name='mid_fc7') \
        .dropout(0.5) \
        .fc(50, relu=False, name='mid_fc8')\
        .get_output()
        #.softmax(name='prob'))
    '''print("HI")
    for op in tf.get_default_graph().get_operations():
       if len(op.outputs) > 0:
           print(str(op.name) + " " + str(op.outputs[0].shape.as_list()))
    sys.exit(0)'''
    return layer

def initialize_imagenet(sess):
    with tf.variable_scope('ResNet18'):
        model_data_path = 'vggm.npy'
        net.load(model_data_path, sess, skip_op=['fc6', 'fc7', 'fc8'])

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
