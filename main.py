#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

import cv2
import tensorflow.python.platform
from types import *
import os
import pickle
from PIL import Image
import jisx0208_2uni

IMAGE_SIZE = 32
CHANNEL = 1
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*CHANNEL

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('readmodels', "models/models.ckpt", 'File name of model data')

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

def nega(img_src):
    img_nega = 255 - img_src
    return img_nega

def binary(img_src):
    bin_npy = np.zeros(img_src.shape, img_src.dtype)
    bin_npy[np.where(img_src > 127)] = 255
    return bin_npy

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

# webapp
app = Flask(__name__)

code_list = None
with open('dataset.pickle', 'rb') as f:
    code_list = pickle.load(f)

NUM_CLASSES = len(code_list)

images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder("float")

logits = inference(images_placeholder, keep_prob, IMAGE_SIZE, CHANNEL, NUM_CLASSES)
sess = tf.InteractiveSession()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess,FLAGS.readmodels)

base = cv2.imread('output.png')

@app.route('/api/handewritten', methods=['POST'])
def handewritten():
    input = ((255 - np.array(request.json, dtype=np.uint8))).reshape(64, 64)
    img2 = np.zeros_like(base)
    img2[:,:,0] = input
    img2[:,:,1] = input
    img2[:,:,2] = input
    img2 = cv2.resize(img2, (IMAGE_SIZE, IMAGE_SIZE))
    img2 = binary(img2)
    img_nega = nega(img2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    input = img2.flatten().astype(np.float32)/255.0
    # print(input)

    pr = logits.eval(feed_dict={
        images_placeholder: [input],
        keep_prob: 1.0 })[0]
    rank = 10
    indexes = get_top_rank_index(pr, rank)
    output1 = []
    output2 = []
    for c in range(rank):
        print("index:"+str(indexes[c])+" code_list[indexes[c]]:"+str(code_list[indexes[c]]))
        output1.append(chr(int(code_list[indexes[c]], 16)))
        output2.append(str(pr[indexes[c]]))

    print(output1)
    print(output2)

    return jsonify(results=[output1, output2])


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
