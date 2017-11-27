import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
import numpy as np
import scipy
import cPickle
import os
from scipy.spatial.distance import cdist
import scipy.io
import sys
# Syspath for the folder with the utils files
#sys.path.insert(0, "/data/sylvestre")

import utils_icarl_core50
import utils_data_core50

######### Modifiable Settings ##########
num_classes = 50
batch_size = 256            # Batch size
nb_groups  = 9              # Number of groups
top        = 1              # Choose to evaluate the top X accuracy
gpu        = '0'            # Used GPU
########################################

######### Paths  ##########
# Working station
execution = sys.argv[1]

devkit_path = '/home/lgatto/core50_batches_filelists/batches_filelists/'+execution
train_path = '/home/admin/core50_128x128'
save_path = '/home/lgatto/core50/savevgg/'+execution+'/'

###########################

file_suffix = execution.replace('/', '')
nb_proto = int(sys.argv[2])
# Load class means
str_class_means = 'class_means'+file_suffix+str(nb_proto)+'.pickle'
with open(str_class_means,'rb') as fp:
      class_means = cPickle.load(fp)

# Loading the labels
files_test, labels_test = utils_data_core50.prepare_test_files(devkit_path)

# Initialization
acc_list = np.zeros((nb_groups,3))

for itera in range(nb_groups):
    print("Processing network after {} increments\t".format(itera))
    
    inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl_core50.reading_data_and_preparing_network(files_test, labels_test, gpu, itera, batch_size, train_path, num_classes, save_path, nb_proto)
    
    with tf.Session(config=config) as sess:
        # Launch the prefetch system
        coord   = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(inits)
        
        # Evaluation routine
        stat_hb1     = []
        stat_icarl = []
        stat_ncm     = []
        
        for i in range(int(np.ceil(len(files_test)/batch_size))):
            sc, l , loss,files_tmp,feat_map_tmp = sess.run([scores, label_batch,loss_class,file_string_batch,op_feature_map])
            mapped_prototypes = feat_map_tmp#[:,0,0,:]
            pred_inter    = (mapped_prototypes.T)/np.linalg.norm(mapped_prototypes.T,axis=0)
            sqd_icarl     = -cdist(class_means[:,:,0,itera].T, pred_inter.T, 'sqeuclidean').T
            sqd_ncm       = -cdist(class_means[:,:,1,itera].T, pred_inter.T, 'sqeuclidean').T
            stat_hb1     += ([ll in best for ll, best in zip(l, np.argsort(sc, axis=1)[:, -top:])])
            stat_icarl   += ([ll in best for ll, best in zip(l, np.argsort(sqd_icarl, axis=1)[:, -top:])])
            stat_ncm     += ([ll in best for ll, best in zip(l, np.argsort(sqd_ncm, axis=1)[:, -top:])])
    
    print('Increment: %i' %itera)
    print('Hybrid 1 top '+str(top)+' accuracy: %f' %np.average(stat_hb1))
    print('iCaRL top '+str(top)+' accuracy: %f' %np.average(stat_icarl))
    print('NCM top '+str(top)+' accuracy: %f' %np.average(stat_ncm))
    acc_list[itera,0] = np.average(stat_icarl)
    acc_list[itera,1] = np.average(stat_hb1)
    acc_list[itera,2] = np.average(stat_ncm)
    
    # Reset the graph to compute the numbers ater the next increment
    tf.reset_default_graph()


np.save('results_top'+str(top)+'_acc__cl',acc_list)
