#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import h5py
import os
from datetime import datetime

def to_h5f(source_filename, output_filename, image_width, image_height, num_classes, chunk, channel=3, test=False):
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    with open(source_filename, 'r') as f:
        if not test:
            h5file = h5py.File(output_filename,'a')
            images = h5file.create_dataset('images', (1, image_width*image_height*channel), dtype='float32', maxshape=(None,None), chunks=(chunk, image_width*image_height*channel))
            labels = h5file.create_dataset('labels', (1, num_classes), dtype='float64', maxshape=(None,None), chunks=(chunk, num_classes))
        count = 0
        image_list = []
        label_list = []
        for line in f:
            line = line.rstrip()
            l = line.split()
            count += 1
            if test:
                print("count:"+str(count) +" "+ l[0])
            if channel is 1:
                img = cv2.imread(l[0], cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(l[0])
            img2 = cv2.resize(img, (image_width, image_height))
            if channel is 1:
                #2値化
                _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)

            if not test:
                image_list.append(img2.flatten().astype(np.float32)/255.0)
                tmp = np.zeros(num_classes)
                tmp[int(l[1])] = 1
                label_list.append(tmp)
            if not test:
                if count % chunk == 0:
                    print(datetime.now().strftime("%Y/%m/%d %H:%M:%S")+' count:'+str(count),flush=True)
                    images.resize(images.shape[0]+chunk, axis=0)
                    images[-chunk:] = np.asarray(images)
                    labels.resize(labels.shape[0]+chunk, axis=0)
                    labels[-chunk:] = np.asarray(label)
                    del image_list[:]
                    del label_list[:]

        if not test:
            if not (count % chunk == 0):
                #ぴったりでなければ残りをくっつける
                remain = count - images.shape[0] + 1
                images.resize(count, axis=0)
                images[-remain:] = np.asarray(image_list)
                labels.resize(count, axis=0)
                labels[-remain:] = np.asarray(label_list)

        if not test:
            h5file.flush()
            h5file.close()

def list2h5f(output_filename, train_images, train_image_labels, test_images, test_image_labels):
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    h5file = h5py.File(output_filename,'a')
    h5file.create_dataset('train_images', data=train_images)
    h5file.create_dataset('train_labels', data=train_image_labels)
    h5file.create_dataset('test_images', data=test_images)
    h5file.create_dataset('test_labels', data=test_image_labels)
    h5file.flush()
    h5file.close()

def load_h5f(source_filename):
    return h5py.File(source_filename, 'r')

if __name__ == '__main__':
    IMAGE_SIZE = 64
    NUM_CLASSES = 113
    to_h5f('huge_train_200.txt', 'huge_train_200.h5', IMAGE_SIZE, IMAGE_SIZE, NUM_CLASSES, 10000, 1, True)


