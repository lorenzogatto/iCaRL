import tensorflow as tf
import numpy as np
import cPickle
import os
import scipy.io
import sys

def read_data(train_path, labels_list, files_from_cl):
    image_list = [train_path + '/' + file_train for file_train in files_from_cl]
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=True, capacity=2000)
    image_file_content = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.resize_images(tf.image.decode_png(image_file_content, channels=3), [128, 128])
    return image, label


def read_data_test(train_path, labels_list, files_from_cl):
    image_list = [train_path + '/' + file_train for file_train in files_from_cl]
    files_list = files_from_cl
    print('read_data_test')
    print(str(len(image_list)) + ' ' +  str(len(files_list)) + ' ' + str(len(labels_list)))
    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    files = tf.convert_to_tensor(files_list, dtype=tf.string)
    labels = tf.convert_to_tensor(labels_list, dtype=tf.int32)
    input_queue = tf.train.slice_input_producer([images, labels, files], shuffle=False, capacity=2000)
    image_file_content = tf.read_file(input_queue[0])
    label = input_queue[1]
    file_string = input_queue[2]
    image = tf.image.resize_images(tf.image.decode_png(image_file_content, channels=3), [128, 128])
    return image, label, file_string

'''
Returns file names and labels for the training data of a certain iteration
'''
def prepare_train_files(train_path, devkit_path, itera):
    filename = devkit_path + '/train_batch_0' + str(itera) + '_filelist.txt'
    file = open(filename, "r")
    files_train = []
    label_train = []
    for line in file:
        files_train.append(line.split(" ")[0])
        label_train.append(int(line.split(" ")[1]))
    return files_train, label_train

'''
Returns file names and labels for the training data of a certain iteration
'''
def prepare_test_files(devkit_path):
    filename = devkit_path + '/test_filelist.txt'
    file = open(filename, "r")
    files_test = []
    label_test = []
    for line in file:
        files_test.append(line.split(" ")[0])
        label_test.append(int(line.split(" ")[1]))
    return files_test, label_test
