import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True #added by me
import numpy as np
import scipy
import pickle as cPickle
import os
import scipy.io
import sys
# Syspath for the folder with the utils files
#sys.path.insert(0, "/media/data/srebuffi")

import utils_resnet
import utils_icarl
import utils_data

######### Modifiable Settings ##########
batch_size = 128            # Batch size 128
nb_val     = 50             # Validation samples per class
nb_cl      = 100            # Classes per group 100
nb_groups  = 10             # Number of groups 10
nb_proto   = 20             # Number of prototypes per class: total protoset memory/ total number of classes
epochs     = 1             # Total number of epochs 60
lr_old     = 2             # Initial learning rate 2
lr_strat   = [20,30,40,50]  # Epochs where learning rate gets decreased
lr_factor  = 5.             # Learning rate decrease factor
gpu        = '0'            # Used GPU
wght_decay = 0.00001        # Weight Decay
########################################

######### Paths  ##########
# Working station 
devkit_path = 'C:\\Users\\gatto\\Desktop\\tesi\\imagenet\\ILSVRC2012_devkit_t12'
train_path  = 'F:\\tesi\\ILSVRC2012_img_train'
save_path   = 'C:\\Users\\gatto\\Desktop\\tesi\\imagenet\\save'

###########################

#####################################################################################################

### Initialization of some variables ###
class_means    = np.zeros((512,nb_groups*nb_cl,2,nb_groups))
loss_batch     = []
files_protoset =[] #prototypes per class. Will contain file names
for _ in range(nb_groups*nb_cl):
    files_protoset.append([])


### Preparing the files for the training/validation ###

# Random mixing
print("Mixing the classes and putting them in batches of classes...")
np.random.seed(1993)
order  = np.arange(nb_groups * nb_cl)
mixing = np.arange(nb_groups * nb_cl)
np.random.shuffle(mixing)

# Loading the labels
labels_dic, label_names, validation_ground_truth = utils_data.parse_devkit_meta(devkit_path)
#labels_dic: dictionary label code to number
#label_names #natural languale names, sorted by class value. not used???
#validation_ground_truth 50k x 1 list of integers (class values) not used???

# Preparing the files per group of classes
print("Creating a validation set ...")
files_train, files_valid = utils_data.prepare_files(train_path, mixing, order, labels_dic, nb_groups, nb_cl, nb_val)

#print("Train files", files_train)
#print("Validation files", files_valid)

# Pickle order and files lists and mixing
with open(str(nb_cl)+'mixing.pickle','wb') as fp:
    cPickle.dump(mixing,fp,protocol=2)

with open(str(nb_cl)+'settings_resnet.pickle','wb') as fp:
    cPickle.dump(order,fp,protocol=2)
    cPickle.dump(files_valid,fp,protocol=2)
    cPickle.dump(files_train,fp,protocol=2)


### Start of the main algorithm ###

for itera in range(nb_groups):
  
  # Files to load : training samples + protoset
  print('Batch of classes number {0} arrives ...'.format(itera+1))
  # Adding the stored exemplars to the training set
  if itera == 0:
    files_from_cl = files_train[itera]
  else:
    files_from_cl = files_train[itera][:]
    for i in range(itera*nb_cl):
      nb_protos_cl  = int(np.ceil(nb_proto*nb_groups*1./itera)) # Reducing number of exemplars of the previous classes
      tmp_var = files_protoset[i]
      files_from_cl += tmp_var[0:min(len(tmp_var),nb_protos_cl)]
  
  ## Import the data reader ##
  image_train, label_train   = utils_data.read_data(train_path, labels_dic, mixing, files_from_cl=files_from_cl)  
  image_batch, label_batch_0 = tf.train.batch([image_train, label_train], batch_size=batch_size, num_threads=8)
  label_batch = tf.one_hot(label_batch_0,nb_groups*nb_cl) #Eg. 4 -> [0, 0, 0, 0, 1]
  
  ## Define the objective for the neural network ##
  if itera == 0:
    # No distillation
    variables_graph,variables_graph2,scores,scores_stored = utils_icarl.prepare_networks(gpu,image_batch, nb_cl, nb_groups)
    
    # Define the objective for the neural network: 1 vs all cross_entropy
    with tf.device('/cpu:0'):
        scores        = tf.concat(scores,0) #puts elements of scores in rows
        l2_reg        = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
        loss_class    = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores)) 
        loss          = loss_class + l2_reg
        learning_rate = tf.placeholder(tf.float32, shape=[])
        opt           = tf.train.MomentumOptimizer(learning_rate, 0.9)
        train_step    = opt.minimize(loss,var_list=variables_graph)
  
  if itera > 0:
    # Distillation
    variables_graph,variables_graph2,scores,scores_stored = utils_icarl.prepare_networks(gpu,image_batch, nb_cl, nb_groups)
    
    # Copying the network to use its predictions as ground truth labels
    op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in range(len(variables_graph))]
    
    # Define the objective for the neural network : 1 vs all cross_entropy + distillation
    with tf.device('/cpu:0'):
      scores            = tf.concat(scores,0)
      scores_stored     = tf.concat(scores_stored,0)
      old_cl            = (order[range(itera*nb_cl)]).astype(np.int32)
      new_cl            = (order[range(itera*nb_cl,nb_groups*nb_cl)]).astype(np.int32)
      label_old_classes = tf.sigmoid(tf.stack([scores_stored[:,i] for i in old_cl],axis=1))
      label_new_classes = tf.stack([label_batch[:,i] for i in new_cl],axis=1)
      pred_old_classes  = tf.stack([scores[:,i] for i in old_cl],axis=1)
      pred_new_classes  = tf.stack([scores[:,i] for i in new_cl],axis=1)
      l2_reg            = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
      loss_class        = tf.reduce_mean(tf.concat([tf.nn.sigmoid_cross_entropy_with_logits(labels=label_old_classes, logits=pred_old_classes),tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes, logits=pred_new_classes)],1))
      loss              = loss_class + l2_reg
      learning_rate     = tf.placeholder(tf.float32, shape=[])
      opt               = tf.train.MomentumOptimizer(learning_rate, 0.9)
      train_step        = opt.minimize(loss,var_list=variables_graph)
  
  ## Run the learning phase ##
  with tf.Session(config=config) as sess:
    # Launch the data reader 
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    sess.run(tf.global_variables_initializer())
    lr      = lr_old
    #TODO initialize the network at first iteration with imagenet pretrain
    # Run the loading of the weights for the learning network and the copy network
    if itera > 0:
      void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
      void1 = sess.run(op_assign)
    
    for epoch in range(epochs):
        print("Batch of classes {} out of {} batches".format(
                itera + 1, nb_groups))
        print('Epoch %i' % epoch)
        print(int(np.ceil(len(files_from_cl)/batch_size)))
        for i in range(int(np.ceil(len(files_from_cl)/batch_size))):
            #print("Cycling batches of training data " + str(i))
            loss_class_val, _ ,sc,lab = sess.run([loss_class, train_step,scores,label_batch_0], feed_dict={learning_rate: lr})
            loss_batch.append(loss_class_val)
            #print("Done " + str(i))
            # Plot the training error every 10 batches
            if len(loss_batch) == 10:
                print(np.mean(loss_batch))
                loss_batch = []
            
            # Plot the training top 1 accuracy every 80 batches
            if (i+1)%80 == 0:
                stat = []
                stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
                stat =np.average(stat)
                print('Training accuracy %f' %stat)
        
        # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
        if epoch in lr_strat:
            lr /= lr_factor
    print("End batch")
    # copy weights to store network
    save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
    utils_resnet.save_model(save_path+'model-iteration'+str(nb_cl)+'-%i.pickle' % itera, scope='ResNet18', sess=sess)
  print("Resetting graph")
  # Reset the graph 
  tf.reset_default_graph()
  
  ## Exemplars management part  ##
  nb_protos_cl  = int(np.ceil(nb_proto*nb_groups*1./(itera+1))) # Reducing number of exemplars for the previous classes
  files_from_cl = files_train[itera]
  print("Preparing network for exemplar management part")
  inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(files_from_cl, gpu, itera, batch_size, train_path, labels_dic, mixing, nb_groups, nb_cl, save_path)
  #inits = weight initializer
  #scores = last layer outputs
  #label_batch =  classes per class
  #loss class= cross-entropy loss
  #op_feature_map = feature map of the input
  print("Done preparing network for exemplar management part")
  with tf.Session(config=config) as sess:
    coord   = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    void3   = sess.run(inits)
    print("Load the training samples of the current batch")
    # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
    #it gets stuck
    Dtot,processed_files,label_dico = utils_icarl.load_class_in_feature_space(files_from_cl, batch_size, scores, label_batch, loss_class, file_string_batch, op_feature_map, sess)
    
    # Herding procedure : ranking of the potential exemplars
    print('Exemplars selection starting ...')
    for iter_dico in range(nb_cl):
        ind_cl     = np.where(label_dico == order[iter_dico+itera*nb_cl])[0]
        D          = Dtot[:,ind_cl]
        files_iter = processed_files[ind_cl]
        mu         = np.mean(D,axis=1)
        w_t        = mu
        step_t     = 0
        while not(len(files_protoset[itera*nb_cl+iter_dico]) == nb_protos_cl) and step_t<1.1*nb_protos_cl:
            tmp_t   = np.dot(w_t,D)
            ind_max = np.argmax(tmp_t)
            w_t     = w_t + mu - D[:,ind_max]
            step_t  += 1
            if files_iter[ind_max] not in files_protoset[itera*nb_cl+iter_dico]:
              files_protoset[itera*nb_cl+iter_dico].append(files_iter[ind_max])
  
  # Reset the graph
  tf.reset_default_graph()
  
  # Class means for iCaRL and NCM 
  print('Computing theoretical class means for NCM and mean-of-exemplars for iCaRL ...')
  for iteration2 in range(itera+1):
      files_from_cl = files_train[iteration2]
      inits,scores,label_batch,loss_class,file_string_batch,op_feature_map = utils_icarl.reading_data_and_preparing_network(files_from_cl, gpu, itera, batch_size, train_path, labels_dic, mixing, nb_groups, nb_cl, save_path)
      
      with tf.Session(config=config) as sess:
          coord   = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)
          void2   = sess.run(inits)
          
          Dtot,processed_files,label_dico = utils_icarl.load_class_in_feature_space(files_from_cl, batch_size, scores, label_batch, loss_class, file_string_batch, op_feature_map, sess)
          
          for iter_dico in range(nb_cl):
              ind_cl     = np.where(label_dico == order[iter_dico+iteration2*nb_cl])[0]
              D          = Dtot[:,ind_cl]
              files_iter = processed_files[ind_cl]
              current_cl = order[range(iteration2*nb_cl,(iteration2+1)*nb_cl)]
              
              # Normal NCM mean
              class_means[:,order[iteration2*nb_cl+iter_dico],1,itera] = np.mean(D,axis=1)
              class_means[:,order[iteration2*nb_cl+iter_dico],1,itera] /= np.linalg.norm(class_means[:,order[iteration2*nb_cl+iter_dico],1,itera])
              
              # iCaRL approximated mean (mean-of-exemplars)
              # use only the first exemplars of the old classes: nb_protos_cl controls the number of exemplars per class
              ind_herding = np.array([np.where(files_iter == files_protoset[iteration2*nb_cl+iter_dico][i])[0][0] for i in range(min(nb_protos_cl,len(files_protoset[iteration2*nb_cl+iter_dico])))])
              D_tmp       = D[:,ind_herding]
              class_means[:,order[iteration2*nb_cl+iter_dico],0,itera] = np.mean(D_tmp,axis=1)
              class_means[:,order[iteration2*nb_cl+iter_dico],0,itera] /= np.linalg.norm(class_means[:,order[iteration2*nb_cl+iter_dico],0,itera])
      
      # Reset the graph
      tf.reset_default_graph()
  
  # Pickle class means and protoset
  with open(str(nb_cl)+'class_means.pickle','wb') as fp:
      cPickle.dump(class_means,fp,protocol=2)
  with open(str(nb_cl)+'files_protoset.pickle','wb') as fp:
      cPickle.dump(files_protoset,fp,protocol=2)


