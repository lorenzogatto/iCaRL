import tensorflow as tf
import numpy as np
import pickle as cPickle
import os
import scipy.io
import sys
import utils_data_core50
import utils_resnet

def reading_data_and_preparing_network(files_from_cl, labels_train, gpu, itera, batch_size, train_path, num_classes, save_path):
    image_train, label_train, file_string       = utils_data_core50.read_data_test(train_path, labels_train, files_from_cl=files_from_cl)
    image_batch, label_batch,file_string_batch = tf.train.batch([image_train, label_train, file_string], batch_size=batch_size, num_threads=8)
    label_batch_one_hot = tf.one_hot(label_batch, num_classes)
    
    ### Network and loss function  
    mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    with tf.variable_scope('ResNet18'):
        with tf.device('/gpu:'+gpu):
            scores         = utils_resnet.ResNet18(image_batch-mean_img, phase='test',num_outputs=num_classes)
            graph          = tf.get_default_graph()
            op_feature_map = graph.get_operation_by_name('ResNet18/pool_last/avg').outputs[0]
    
    loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch_one_hot, logits=scores))
    
    ### Initilization
    params = dict(cPickle.load(open(save_path+'model-iteration-%i.pickle' % itera, "rb")))
    inits  = utils_resnet.get_weight_initializer(params)
    
    return inits,scores,label_batch,loss_class,file_string_batch,op_feature_map


def reading_data_and_preparing_network_old(files_from_cl, gpu, itera, batch_size, train_path, nb_groups, nb_cl, save_path):
    image_train, label_train, file_string = utils_data.read_data_test(train_path, labels_dic, mixing,
                                                                      files_from_cl=files_from_cl)
    image_batch, label_batch, file_string_batch = tf.train.batch([image_train, label_train, file_string],
                                                                 batch_size=batch_size, num_threads=8)
    label_batch_one_hot = tf.one_hot(label_batch, nb_groups * nb_cl)

    ### Network and loss function
    mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    with tf.variable_scope('ResNet18'):
        with tf.device('/gpu:' + gpu):
            scores = utils_resnet.ResNet18(image_batch - mean_img, phase='test', num_outputs=nb_cl * nb_groups)
            graph = tf.get_default_graph()
            op_feature_map = graph.get_operation_by_name('ResNet18/pool_last/avg').outputs[0]

    loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch_one_hot, logits=scores))

    ### Initilization
    params = dict(cPickle.load(open(save_path + 'model-iteration' + str(nb_cl) + '-%i.pickle' % itera, "rb")))
    inits = utils_resnet.get_weight_initializer(params)

    return inits, scores, label_batch, loss_class, file_string_batch, op_feature_map

def load_class_in_feature_space(files_from_cl,batch_size,scores, label_batch,loss_class,file_string_batch,op_feature_map,sess):
    processed_files=[]
    label_dico=[]
    Dtot=[]
    #print("test", files_from_cl, batch_size)
    print("batches ", int(np.ceil(len(files_from_cl)/batch_size)+1))
    for i in range(int(np.ceil(len(files_from_cl)/batch_size)+1)):
        print(i)#, files_from_cl)
        #sess.run([scores])
        #print(i)
        sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])
        print("done")
        processed_files.extend(files_tmp)
        label_dico.extend(l)
        mapped_prototypes = feat_map_tmp[:,0,0,:]
        Dtot.append((mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0))
    print("test_")
    Dtot            = np.concatenate(Dtot,axis=1)
    processed_files = np.array(processed_files)
    label_dico      = np.array(label_dico)
    return Dtot,processed_files,label_dico

'''
Returns two equivalent resnet networks
'''
#the second one is used as backup of first one to calculate old sigmoid values
def prepare_networks(gpu,image_batch, nb_classes):
  mean_img = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  scores   = []
  with tf.variable_scope('ResNet18'):
    with tf.device('/gpu:' + gpu):
        score = utils_resnet.ResNet18(image_batch-mean_img, phase='train',num_outputs=nb_classes)
        scores.append(score)
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
  
  # First score and initialization
  variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='ResNet18')
  scores_stored   = []
  with tf.variable_scope('store_ResNet18'):
    with tf.device('/gpu:' + gpu):
        score = utils_resnet.ResNet18(image_batch-mean_img, phase='test',num_outputs=nb_classes)
        scores_stored.append(score)
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
  
  variables_graph2 = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='store_ResNet18')
  
  return variables_graph,variables_graph2,scores,scores_stored

