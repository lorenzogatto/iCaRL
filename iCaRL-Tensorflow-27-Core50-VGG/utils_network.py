'''
    This module dynamically loads the selected network based on running configuration.
'''
from config import *

if network == 'mvggnet':
    import mvgg as net
else:
    import mcaffenet as net

def create_model(inp, phase, num_outputs=1000, alpha=0.0):
    return net.create_model(inp, phase, num_outputs, alpha)

def initialize_imagenet(sess):
     net.initialize_imagenet(sess)

def get_weight_initializer(params):
     return net.get_weight_initializer(params)

def save_model(name, scope, sess):
     net.save_model(name, scope, sess)
