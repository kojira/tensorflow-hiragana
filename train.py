#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from datetime import datetime
import gc
import os
import h5f_util
from os import listdir
from os.path import isfile, join
from skimage.io import imread
from scipy.misc import imresize
import random
import pickle


IMAGE_SIZE = 32
CHANNEL = 1
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*CHANNEL

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('test_file', None, 'file name of test image')
flags.DEFINE_string('load_image_dir', 'unicodes/', 'dir name of train images')
flags.DEFINE_string('save_model_dir', 'models/', 'dir name of save model data')
flags.DEFINE_string('out_dataset', 'dataset', 'file name of output dataset file')
flags.DEFINE_string('restore_model', None, 'restore model file path')
flags.DEFINE_integer('max_steps', 20, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 64, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

random.seed(1)


def load_images(root, test_rate=0.2, image_size=IMAGE_SIZE):
    train_images = []
    train_image_indexes = []
    test_images = []
    test_image_indexes = []
    invalid_count = 0
    _class_list = listdir(root)
    except_list = []
    for filename in _class_list:
        _path = join(root, filename)
        if not os.path.isdir(_path):
            except_list.append(filename)

    for filename in except_list:
        _class_list.remove(filename)

    classes = len(_class_list)

    for i, subdir in enumerate(_class_list):
        _path = join(root, subdir)
        print(_path)
        imgs = listdir(_path)
        class_ix = _class_list.index(subdir)
        one_class_images = []
        one_class_indexes = []
        for j, img_name in enumerate(imgs):
            img_arr = imread(join(root, subdir, img_name),as_grey=True)
            img_arr_rs = img_arr
            try:
                img_arr_rs = imresize(img_arr, (image_size, image_size), interp='nearest')
                img_arr_rs = img_arr_rs.flatten().astype('float32')
                img_arr_rs /= 255.0
                one_class_images.append(img_arr_rs)
                tmp = np.zeros(classes)
                tmp[class_ix] = 1
                one_class_indexes.append(tmp)
            except Exception as e:
                print('Skipping bad image: ', e, subdir, img_name)
                invalid_count += 1

        class_image_count = len(one_class_images)
        train_images.extend(one_class_images[:int((class_image_count*(1.0-test_rate)))])
        train_image_indexes.extend(one_class_indexes[:int((class_image_count*(1.0-test_rate)))])
        test_images.extend(one_class_images[int((class_image_count*(1.0-test_rate)))+1:])
        test_image_indexes.extend(one_class_indexes[int((class_image_count*(1.0-test_rate)))+1:])

    print(classes, 'classes')
    print(len(train_images)+len(test_images), 'images loaded')
    print(invalid_count, 'images skipped')
    return np.array(train_images), np.array(train_image_indexes), np.array(test_images), np.array(test_image_indexes), _class_list

def open_image_file(base_dir, code):
    _path = join(base_dir, code)
    image_files = os.listdir(_path)
    for image_file in image_files:
        if not image_file.startswith('.'):
            # print(image_file)
            os.system('open '+_path+'/'+image_file)
            break

def get_top_rank_index(src_array, K):
    # ソートはされていない上位k件のインデックス
    unsorted_max_indices = np.argpartition(-src_array, K)[:K]
    # 上位k件の値
    y = src_array[unsorted_max_indices]
    # 大きい順にソートし、インデックスを取得
    indices = np.argsort(-y)
    # 類似度上位k件のインデックス
    max_k_indices = unsorted_max_indices[indices]
    return max_k_indices


def inference(images_placeholder, keep_prob, imageSize, channel, numClasses):
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    # 入力をimageSize x imageSize x channelに変形
    x_images = tf.reshape(images_placeholder, [-1, imageSize, imageSize, channel])
    # 畳み込み層1
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, channel, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_images, W_conv1) + b_conv1)
    # プーリング層1
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    # 畳み込み層2
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # プーリング層2
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    # 全結合層1
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([(imageSize//4)*(imageSize//4)*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, (imageSize//4)*(imageSize//4)*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    # 全結合層2
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, numClasses])
        b_fc2 = bias_variable([numClasses])
    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    return y_conv

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S")+' train image loading...',flush=True)

    if not os.path.isfile(FLAGS.out_dataset+".h5"):
        train_image, train_label, test_image, test_label, class_list = load_images(FLAGS.load_image_dir)
        nb_classes = len(class_list)
        h5f_util.list2h5f(FLAGS.out_dataset+".h5", train_image, train_label, test_image, test_label)
        with open(FLAGS.out_dataset+".pickle", 'wb') as f:
            pickle.dump(class_list, f)
    else:
        print("loading hdf5 dataset...")
        dataset = h5f_util.load_h5f(FLAGS.out_dataset+".h5")
        train_image = np.array(dataset['train_images'])
        train_label = np.array(dataset['train_labels'])
        test_image = np.array(dataset['test_images'])
        test_label = np.array(dataset['test_labels'])
        print("done.")
        nb_classes = test_label.shape[1]
        with open(FLAGS.out_dataset+".pickle", 'rb') as f:
            class_list = pickle.load(f)

    combined = list(zip(train_image, train_label))
    random.shuffle(combined)
    train_image[:], train_label[:] = zip(*combined)

    print("classes:",nb_classes)

    train_len = len(train_image)
    test_len = len(test_image)

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        labels_placeholder = tf.placeholder("float", shape=(None, nb_classes))
        keep_prob = tf.placeholder("float")

        logits = inference(images_placeholder, keep_prob, IMAGE_SIZE, CHANNEL, nb_classes)

        if not (FLAGS.test_file is None):
            if not (FLAGS.restore_model is None):
                #復元
                sess = tf.InteractiveSession()
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.restore(sess, FLAGS.restore_model)
                img = cv2.imread(FLAGS.test_file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                #二値化
                _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
                test_array = img.flatten().astype(np.float32)/255.0
                pr = logits.eval(feed_dict={ 
                    images_placeholder: [test_array],
                    keep_prob: 1.0 })[0]
                pred = np.argmax(pr)
                open_image_file(FLAGS.load_image_dir, class_list[pred])
                rank = 10
                indexes = get_top_rank_index(pr, rank)
                outputs = []
                outputs2 = []
                for c in range(rank):
                    outputs.append(chr(int(class_list[indexes[c]], 16)))
                    outputs2.append(str(pr[indexes[c]]))
                print(outputs)
                print(outputs2)
            else:
                print("Need restore model path...")
            exit()

        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, FLAGS.learning_rate)
        acc = accuracy(logits, labels_placeholder)

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()

        # 訓練の実行
        if train_len % FLAGS.batch_size is 0:
            train_batch = train_len//FLAGS.batch_size
        else:
            train_batch = (train_len//FLAGS.batch_size)+1
            print("train_batch = "+str(train_batch))
        
        if not (FLAGS.restore_model is None):
            #復元
            saver.restore(sess, FLAGS.restore_model)

        _step = 0
        try:
            for step in range(FLAGS.max_steps):
                _step = step
                for i in range(train_batch):
                    batch = FLAGS.batch_size*i
                    batch_plus = FLAGS.batch_size*(i+1)
                    if batch_plus > train_len:
                        batch_plus = train_len

                    sess.run(train_op, feed_dict={
                      images_placeholder: train_image[batch:batch_plus],
                      labels_placeholder: train_label[batch:batch_plus],
                      keep_prob: 0.5})

                if step % 5 == 0:
                    train_accuracy = 0.0
                    for i in range(train_batch):
                        batch = FLAGS.batch_size*i
                        batch_plus = FLAGS.batch_size*(i+1)
                        if batch_plus > train_len: batch_plus = train_len
                        train_accuracy += sess.run(acc, feed_dict={
                            images_placeholder: train_image[batch:batch_plus],
                            labels_placeholder: train_label[batch:batch_plus],
                            keep_prob: 1.0})
                        if i is not 0: train_accuracy /= 2.0
                    print("%s step %d, training accuracy %g" % (datetime.now().strftime("%Y/%m/%d %H:%M:%S"),step, train_accuracy),flush=True)
        except KeyboardInterrupt:
            os.makedirs(FLAGS.save_model_dir, exist_ok=True)
            save_path = saver.save(sess, FLAGS.save_model_dir+'interrupted_'+str(_step)+'epoc.ckpt')
            exit()

    if test_len % FLAGS.batch_size is 0:
        test_batch = test_len//FLAGS.batch_size
    else:
        test_batch = (test_len//FLAGS.batch_size)+1
        print("test_batch = "+str(test_batch),flush=True)
    test_accuracy = 0.0
    for i in range(test_batch):
        batch = FLAGS.batch_size*i
        batch_plus = FLAGS.batch_size*(i+1)
        if batch_plus > train_len: batch_plus = train_len
        test_accuracy += sess.run(acc, feed_dict={
                images_placeholder: test_image[batch:batch_plus],
                labels_placeholder: test_label[batch:batch_plus],
                keep_prob: 1.0})
        if i is not 0: test_accuracy /= 2.0
    print("test accuracy %g"%(test_accuracy),flush=True)
    os.makedirs(FLAGS.save_model_dir, exist_ok=True)
    save_path = saver.save(sess, FLAGS.save_model_dir+"models.ckpt")
