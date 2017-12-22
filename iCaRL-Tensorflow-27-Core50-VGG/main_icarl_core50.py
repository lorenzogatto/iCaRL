import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
import numpy as np
import scipy
import cPickle
import os
import scipy.io
import sys
import argparse

import utils_network
import utils_icarl_core50
import utils_data_core50
import sys
from config import *

######### Paths  ##########
# Working station
devkit_path = '/home/lgatto/core50_batches_filelists/batches_filelists/'+execution  # where batches are defined
train_path = '/home/admin/core50_128x128'
save_path = '/home/lgatto/core50/savevgg/'+execution+'/' # will store files needed for testing

### Initialization of some variables ###
loss_batch = []
class_means = np.zeros((2048, num_classes, 2, nb_batches))
files_protoset = []  # prototypes per class. Will contain file names
for _ in range(num_classes):
    files_protoset.append([])

# Preparing the files per group of classes
files_train = [None for i in range(nb_batches)]
labels_train = [None for i in range(nb_batches)]

tf.set_random_seed(1)
np.random.seed(1)

for itera in range(nb_batches):
    files_train[itera], labels_train[itera] = utils_data_core50.prepare_train_files(train_path, devkit_path, itera)

### Start of the main algorithm ###
for itera in range(nb_batches):
    labels_from_cl = labels_train[itera][:]

    # Files to load: training samples + protoset
    print('Batch of classes number {0} arrives ...'.format(itera + 1))
    # Adding the stored exemplars to the training set
    if itera == 0:
        files_from_cl = files_train[itera]
    else:
        files_from_cl = files_train[itera][:]
        for i in range(num_classes_itera[itera]):
            nb_protos_cl = int(
                np.ceil(nb_proto * nb_batches * 1. / itera))  # Reducing number of exemplars of the previous classes
            tmp_var = files_protoset[i]
            files_from_cl += tmp_var[0:min(len(tmp_var), nb_protos_cl)]
            labels_from_cl += [i for x in range(min(len(tmp_var), nb_protos_cl))]

    ## Import the data reader ##
    image_train, labels_train_tensor = utils_data_core50.read_data(train_path, labels_from_cl,
                                                                   files_from_cl=files_from_cl)
    image_batch, label_batch_0 = tf.train.batch([image_train, labels_train_tensor], batch_size=batch_size,
                                                num_threads=8)
    label_batch = tf.one_hot(label_batch_0, num_classes_itera[nb_batches])  # Eg. 4 -> [0, 0, 0, 0, 1]

    ## Define the objective for the neural network ##
    if itera == 0:
        # No distillation
        variables_graph, variables_graph2, scores, scores_stored = utils_icarl_core50.prepare_networks(gpu, image_batch,
                                                                                                       num_classes)
        # Define the objective for the neural network: 1 vs all cross_entropy
        with tf.device('/gpu:' + gpu):
            scores = tf.concat(scores, 0)  # puts elements of scores in rows
            l2_reg = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
            loss_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_batch, logits=scores))
            loss = loss_class + l2_reg
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.MomentumOptimizer(learning_rate, momentum)
            train_step = opt.minimize(loss, var_list=variables_graph)

    if itera > 0:
        # Distillation
        variables_graph, variables_graph2, scores, scores_stored = utils_icarl_core50.prepare_networks(gpu, image_batch,
                                                                                                num_classes)
        # Copying the network to use its predictions as ground truth labels
        op_assign = [(variables_graph2[i]).assign(variables_graph[i]) for i in range(len(variables_graph))]

        # Define the objective for the neural network : 1 vs all cross_entropy + distillation
        with tf.device('/gpu:' + gpu):
            scores = tf.concat(scores, 0)
            scores_stored = tf.concat(scores_stored, 0)
            old_cl = range(num_classes_itera[itera])
            new_cl = range(num_classes_itera[itera], num_classes_itera[nb_batches])

            label_old_classes = tf.sigmoid(tf.stack([scores_stored[:, i] for i in old_cl], axis=1))
            label_new_classes = tf.stack([label_batch[:, i] for i in new_cl], axis=1)
            pred_old_classes = tf.stack([scores[:, i] for i in old_cl], axis=1)
            pred_new_classes = tf.stack([scores[:, i] for i in new_cl], axis=1)
            l2_reg = wght_decay * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='ResNet18'))
            loss_class = tf.reduce_mean(tf.concat(
                [tf.nn.sigmoid_cross_entropy_with_logits(labels=label_old_classes, logits=pred_old_classes),
                 tf.nn.sigmoid_cross_entropy_with_logits(labels=label_new_classes, logits=pred_new_classes)], 1))
            loss = loss_class + l2_reg
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt = tf.train.MomentumOptimizer(learning_rate, momentum)
            train_step = opt.minimize(loss, var_list=variables_graph)
    print('Length of files_from_cl: ' + str(len(files_from_cl)))
    ## Run the learning phase ##
    with tf.Session(config=config) as sess:
        # Launch the data reader
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())

        # Initialize the network at first iteration with imagenet pretrain
        if itera == 0:
            utils_network.initialize_imagenet(sess)
            current_lr = initial_lr_first_batch
            current_batch_epochs = epochs_first_batch
        # Run the loading of the weights for the learning network and the copy network
        if itera > 0:
            void0 = sess.run([(variables_graph[i]).assign(save_weights[i]) for i in range(len(variables_graph))])
            void1 = sess.run(op_assign)
            current_lr = initial_lr_other_batches
            current_batch_epochs = epochs_other_batches

        for epoch in range(current_batch_epochs):
            # Decrease the learning by 5 every 10 epoch after 20 epochs at the first learning rate
            if (itera == 0 and epoch*2 in lr_strat_first_batch) or (itera != 0 and epoch in lr_strat_other_batches):
                current_lr /= lr_factor

            print("Batch of classes {} out of {} batches".format(
                itera + 1, nb_batches))
            print('Epoch %i' % epoch)
            print(int(np.ceil(len(files_from_cl) / batch_size)))
            for i in range(int(np.ceil(len(files_from_cl) / batch_size))):
                # print("Cycling batches of training data " + str(i))
                loss_class_val, _, sc, lab = sess.run([loss_class, train_step, scores, label_batch_0],
                                                      feed_dict={learning_rate: current_lr})
                loss_batch.append(loss_class_val)
                # Plot the training error every 10 batches
                if len(loss_batch) == 10:
                    print(np.mean(loss_batch))
                    loss_batch = []

                # Plot the training top 1 accuracy every 80 batches
                if (i + 1) % 40 == 0:
                    stat = []
                    stat += ([ll in best for ll, best in zip(lab, np.argsort(sc, axis=1)[:, -1:])])
                    stat = np.average(stat)
                    print('Training accuracy %f' % stat)

        print("End batch")
        # copy weights to store network
        save_weights = sess.run([variables_graph[i] for i in range(len(variables_graph))])
        utils_network.save_model(save_path + str(nb_proto) + 'model-iteration-%i.pickle' % itera, scope='ResNet18', sess=sess)
    print("Resetting graph")
    # Reset the graph
    tf.reset_default_graph()

    ## Exemplars management part  ##
    nb_protos_cl = int(
        np.ceil(nb_proto * nb_batches * 1. / (itera + 1)))  # Reducing number of exemplars for the previous classes

    files_from_cl = files_train[itera]
    labels_from_cl = labels_train[itera]
    print("Preparing network for exemplar management part")
    inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_icarl_core50.reading_data_and_preparing_network(
        files_from_cl, labels_from_cl, gpu, itera, batch_size, train_path, num_classes, save_path, nb_proto)
    # inits = weight initializer
    # scores = last layer outputs
    # loss class= cross-entropy loss
    # op_feature_map = feature map of the input
    print("Done preparing network for exemplar management part")
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        void3 = sess.run(inits)
        print("Load the training samples of the current batch")
        # Load the training samples of the current batch of classes in the feature space to apply the herding algorithm
        # it gets stuck
        Dtot, processed_files, label_dico = utils_icarl_core50.load_class_in_feature_space(files_from_cl, batch_size,
                                                                                           scores, label_batch,
                                                                                           loss_class,
                                                                                           file_string_batch,
                                                                                           op_feature_map, sess)
        # Dtot = # train images * classes matrix
        # label_dico labels for train images
        # Herding procedure : ranking of the potential exemplars
        print('Exemplars selection starting ...')
        for iter_dico in range(num_classes_itera[itera + 1] - num_classes_itera[itera]):
            ind_cl = np.where(label_dico == iter_dico + num_classes_itera[itera])[0]
            D = Dtot[:, ind_cl]
            files_iter = processed_files[ind_cl]
            mu = np.mean(D, axis=1)
            w_t = mu
            step_t = 0
            while not (len(files_protoset[num_classes_itera[
                itera] + iter_dico]) == nb_protos_cl) and step_t < 1.1 * nb_protos_cl:
                tmp_t = np.dot(w_t, D)
                ind_max = np.argmax(tmp_t)
                w_t = w_t + mu - D[:, ind_max]
                step_t += 1
                if files_iter[ind_max] not in files_protoset[num_classes_itera[itera] + iter_dico]:
                    files_protoset[num_classes_itera[itera] + iter_dico].append(files_iter[ind_max])

    # Reset the graph
    tf.reset_default_graph()

    # Class means for iCaRL and NCM
    print('Computing theoretical class means for NCM and mean-of-exemplars for iCaRL ...')
    for iteration2 in range(itera + 1):
        files_from_cl = files_train[iteration2]
        labels_from_cl = labels_train[iteration2]
        #print('Files from cl' + str(files_from_cl));
        #print('labels_from_cl' + labels_from_cl);
        inits, scores, label_batch, loss_class, file_string_batch, op_feature_map = utils_icarl_core50.reading_data_and_preparing_network(
            files_from_cl, labels_from_cl, gpu, itera, batch_size, train_path, num_classes, save_path, nb_proto)

        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            void2 = sess.run(inits)

            Dtot, processed_files, label_dico = utils_icarl_core50.load_class_in_feature_space(files_from_cl,
                                                                                               batch_size, scores,
                                                                                               label_batch, loss_class,
                                                                                               file_string_batch,
                                                                                               op_feature_map, sess)

            for iter_dico in range(num_classes_itera[iteration2 + 1] - num_classes_itera[iteration2]):
                ind_cl = np.where(label_dico == iter_dico + num_classes_itera[iteration2])[0]
                D = Dtot[:, ind_cl]
                files_iter = processed_files[ind_cl]
                current_cl = range(num_classes_itera[iteration2], num_classes_itera[iteration2 + 1])

                # Normal NCM mean
                class_means[:, num_classes_itera[iteration2] + iter_dico, 1, itera] = np.mean(D, axis=1)
                class_means[:, num_classes_itera[iteration2] + iter_dico, 1, itera] /= np.linalg.norm(
                    class_means[:, num_classes_itera[iteration2] + iter_dico, 1, itera])

                # iCaRL approximated mean (mean-of-exemplars)
                # use only the first exemplars of the old classes: nb_protos_cl controls the number of exemplars per class
                ind_herding = np.array(
                    [np.where(files_iter == files_protoset[num_classes_itera[iteration2] + iter_dico][i])[0][0] for i in
                     range(min(nb_protos_cl, len(files_protoset[num_classes_itera[iteration2] + iter_dico])))])
                D_tmp = D[:, ind_herding]
                class_means[:, num_classes_itera[iteration2] + iter_dico, 0, itera] = np.mean(D_tmp, axis=1)
                class_means[:, num_classes_itera[iteration2] + iter_dico, 0, itera] /= np.linalg.norm(
                    class_means[:, num_classes_itera[iteration2] + iter_dico, 0, itera])

        # Reset the graph
        tf.reset_default_graph()

    file_suffix = execution.replace('/', '')
    # Pickle class means and protoset
    with open('outputs/class_means'+file_suffix+str(nb_proto)+'.pickle', 'wb') as fp:
        cPickle.dump(class_means, fp, protocol=2)
    with open('outputs/files_protoset'+file_suffix+str(nb_proto)+'.pickle', 'wb') as fp:
        cPickle.dump(files_protoset, fp, protocol=2)
