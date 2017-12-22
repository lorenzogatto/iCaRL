import tensorflow as tf
import numpy as np
import cPickle
import os
import scipy.io
import sys
import utils_data_core50
import utils_network

'''
    Reads data and loads the trained network
'''
def reading_data_and_preparing_network(files_from_cl, labels_train, gpu, itera, batch_size, train_path, num_classes, save_path, nb_proto):
    image_train, label_train, file_string       = utils_data_core50.read_data_test(train_path, labels_train, files_from_cl=files_from_cl)
    image_batch, label_batch,file_string_batch = tf.train.batch([image_train, label_train, file_string], batch_size=batch_size, num_threads=8)
    label_batch_one_hot = tf.one_hot(label_batch, num_classes)
    
    ### Network and loss function  
    mean_img = tf.constant([123, 117, 104], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    with tf.variable_scope('ResNet18'):
        with tf.device('/gpu:'+gpu):
            scores         = utils_network.create_model(image_batch - mean_img, phase='test', num_outputs=num_classes)
            graph          = tf.get_default_graph()
            #ops = graph.get_operations()
            #for op in ops:
            #    print(op.name + "\n")
            op_feature_map = graph.get_operation_by_name('ResNet18/mid_fc7/ResNet18/mid_fc7').outputs[0]
    
    loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch_one_hot, logits=scores))
    print(save_path+str(nb_proto)+'model-iteration-%i.pickle' % itera)
    ### Initialization
    params = dict(cPickle.load(open(save_path+str(nb_proto)+'model-iteration-%i.pickle' % itera, "r")))
    inits  = utils_network.get_weight_initializer(params)
    return inits,scores,label_batch,loss_class,file_string_batch,op_feature_map

def load_class_in_feature_space(files_from_cl,batch_size,scores, label_batch,loss_class,file_string_batch,op_feature_map,sess):
    processed_files=[]
    label_dico=[]
    Dtot=[]
    #print("test", files_from_cl, batch_size)
    print("batches ", int(np.ceil(len(files_from_cl)/batch_size)+1))
    for i in range(int(np.ceil(len(files_from_cl)/batch_size)+1)):
        #print(i)#, files_from_cl)
        #sess.run([scores])
        #print(i)
        sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])
        #print("done")
        processed_files.extend(files_tmp)
        label_dico.extend(l)
        mapped_prototypes = feat_map_tmp
        Dtot.append((mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0))
    #print("test_")
    Dtot            = np.concatenate(Dtot,axis=1)
    processed_files = np.array(processed_files)
    label_dico      = np.array(label_dico)
    return Dtot,processed_files,label_dico

'''
    Returns two equivalent networks
'''
# The second network is used as backup of first one to calculate old sigmoid values
def prepare_networks(gpu, image_batch, nb_classes):
  mean_img = tf.constant([123, 117, 104], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
  scores   = []
  with tf.variable_scope('ResNet18'):
    with tf.device('/gpu:' + gpu):
        score = utils_network.create_model(image_batch - mean_img, phase='train', num_outputs=nb_classes)
        scores.append(score)
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
  
  # First score and initialization
  variables_graph = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='ResNet18')
  '''for v in variables_graph:
      print(v.name)'''
  scores_stored   = []
  with tf.variable_scope('store_ResNet18'):
    with tf.device('/gpu:' + gpu):
        score = utils_network.create_model(image_batch - mean_img, phase='test', num_outputs=nb_classes)
        scores_stored.append(score)
    
    scope = tf.get_variable_scope()
    scope.reuse_variables()
  
  variables_graph2 = tf.get_collection(tf.GraphKeys.WEIGHTS, scope='store_ResNet18')

  return variables_graph,variables_graph2,scores,scores_stored


