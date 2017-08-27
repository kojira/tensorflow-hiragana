#!/usr/bin/env python
# -*- coding: utf-8 -*-
import struct
from PIL import Image, ImageOps
import os
import jisx0208_2uni
import numpy as np
import cv2
import sys

def tobinary(img_src):
    _, bin_cv2 = cv2.threshold(img_src, 0, 255, cv2.THRESH_OTSU)
    return bin_cv2

filename_base = 'ETL8G/ETL8G_'
id_record = 0
sz_record = 8199

args = sys.argv

only_one = False

if (len(args) > 1):
    only_one = True if int(args[1]) > 0 else False

only_hiragana = False

if (len(args) > 2):
    only_hiragana = True if int(args[2]) > 0 else False

done_list = []

hiraganas = "あいうえおかきくけこがぎぐげごさしすせそざじずぜぞたちつてとだぢづでどなにぬねのはひふへほばびぶべぼぱぴぷぺぽまみむめもやゆよらりるれろわをん"


for dataset_id in range(1, 34):
    filename = filename_base + '{:02d}'.format(dataset_id)

    with open(filename, 'rb') as f:
        for id_record in range(956):
            print(filename + ":"+str(id_record * sz_record))
            f.seek(id_record * sz_record)
            s = f.read(sz_record)
            r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
            iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
            code = jisx0208_2uni.jisx0208_2uni(r[1])
            if only_one:
                if code in done_list:
                    continue
            if only_hiragana:
                print("code:"+str(code)+" :"+chr(code))
                if not (chr(code) in hiraganas):
                # if not (b'.HIRA' in r[2]):
                    continue
            code_str = "{:04x}".format(code)[-4:]
            directory = 'unicodes/{:s}/'.format(code_str)
            # if os.path.exists(directory):
            #     if len(os.listdir(directory)) >= 100:
            #         continue
            fn = 'ETL9G_{:02d}_{:d}_{:s}.png'.format(dataset_id,(r[0]-1)%20+1, hex(code)[-4:])
            #iP.save(fn, 'PNG', bits=4)
            iP = iF.convert('RGB')
            iP = iP.resize((64,64), Image.ANTIALIAS)
            iP = ImageOps.autocontrast(iP, 0)
            iP = iP.convert('P')
            im_numpy = tobinary(np.array(iP))
            iE = Image.fromarray(im_numpy)
            if not os.path.exists(directory):
                os.makedirs(directory)
            iE.save(directory+fn, 'PNG')
            done_list.append(code)
    print(filename + " done.")
