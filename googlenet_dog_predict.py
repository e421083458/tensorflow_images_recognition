# -*- coding: utf-8 -*-

""" GoogLeNet.
Applying 'GoogLeNet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Szegedy, Christian, et al.
    Going deeper with convolutions.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [GoogLeNet Paper](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import os
import tflearn.datasets.oxflower17 as oxflower17
import argparse
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import scipy
import numpy as np
from tflearn.data_utils import *
import sys

def main(_):
    #X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

    network = input_data(shape=[None, 227, 227, 3])
    conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


    inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


    inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

    inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
    inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
    inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
    inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
    inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

    inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
    inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
    inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
    inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


    inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
    inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


    inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
    inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

    pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)
    loss = fully_connected(pool5_7_7, 12,activation='softmax')
    network = regression(loss, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    model = tflearn.DNN(network, checkpoint_path='model_dog',
                        max_checkpoints=1, tensorboard_verbose=2)
    # model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
    #           show_metric=True, batch_size=64, snapshot_step=200,
    #           snapshot_epoch=False, run_id='googlenet_oxflowers17')
    model_path = os.path.join(FLAGS.checkpointDir, "model_dog.tfl")
    print(model_path)
    model.load(model_path)

    # predict_pic = os.path.join(FLAGS.buckets, "bird_mount_bluebird.jpg")
    # file_paths = tf.train.match_filenames_once(predict_pic)
    # input_file_queue = tf.train.string_input_producer(file_paths)
    # reader = tf.WholeFileReader()
    # file_path, raw_data = reader.read(input_file_queue)
    # img = tf.image.decode_jpeg(raw_data, 3)
    # img = tf.image.resize_images(img, [32, 32])
    # prediction = model.predict([img])
    # print (prediction[0])
    # predict_pic = os.path.join("./", "image_6.jpg")
    # img_obj = file_io.read_file_to_string(predict_pic)
    # file_io.write_string_to_file("image_47.jpg", img_obj)
    # img = scipy.ndimage.imread("image_6.jpg", mode="RGB")
    # img = scipy.misc.imresize(img, (227, 227), interp="bicubic").astype(np.float32, casting='unsafe')
    im = load_image("image_6.jpg")
    im = resize_image(im, 227, 227)
    im = pil_to_nparray(im)
    im /= 255.

    # Predict
    prediction = model.predict([im])
    predict_label = model.predict_label([im])
    print(prediction)
    print(predict_label)
    print(max(prediction[0]))
    print (prediction[0].tolist().index(max(prediction[0])))
    label_list = range(12)
    label_list[0] = "博美犬"
    label_list[1] = "吉娃娃"
    label_list[2] = "哈士奇"
    label_list[3] = "德国牧羊犬"
    label_list[4] = "杜宾犬"
    label_list[5] = "松狮"
    label_list[6] = "柴犬"
    label_list[7] = "秋田犬"
    label_list[8] = "藏獒"
    label_list[9] = "蝴蝶犬"
    label_list[10] = "贵宾犬"
    label_list[11] = "边境牧羊犬"
    print ("This is a %s"%(label_list[prediction[0].tolist().index(max(prediction[0]))]))

    # print ("This is a %s"%(num[prediction[0].tolist().index(max(prediction[0]))]))
    # predict_pic = os.path.join(FLAGS.buckets, "bird_mount_bluebird.jpg")
    # img = scipy.ndimage.imread(predict_pic, mode="RGB")
    # img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
    # prediction = model.predict([img])
    #print (prediction[0])
    # label_list len=176
    # label_list[0]="dog_images/jpg/万能梗"
    # label_list[1]="dog_images/jpg/丝毛梗"
    # label_list[2]="dog_images/jpg/中国冠毛犬"
    # label_list[3]="dog_images/jpg/中国沙皮犬"
    # label_list[4]="dog_images/jpg/伊比赞猎犬"
    # label_list[5]="dog_images/jpg/伯恩山犬"
    # label_list[6]="dog_images/jpg/兰波格犬"
    # label_list[7]="dog_images/jpg/冰岛牧羊犬"
    # label_list[8]="dog_images/jpg/凯利蓝梗"
    # label_list[9]="dog_images/jpg/凯斯梗"
    # label_list[10]="dog_images/jpg/切萨皮克海湾寻回犬"
    # label_list[11]="dog_images/jpg/刚毛猎狐梗"
    # label_list[12]="dog_images/jpg/匈牙利牧羊犬"
    # label_list[13]="dog_images/jpg/北京犬"
    # label_list[14]="dog_images/jpg/博伊金猎犬"
    # label_list[15]="dog_images/jpg/博得猎狐犬"
    # label_list[16]="dog_images/jpg/博美犬"
    # label_list[17]="dog_images/jpg/卡斯罗"
    # label_list[18]="dog_images/jpg/卡迪根威尔士柯基犬"
    # label_list[19]="dog_images/jpg/卷毛寻回犬"
    # label_list[20]="dog_images/jpg/卷毛比雄犬"
    # label_list[21]="dog_images/jpg/古代英国牧羊犬"
    # label_list[22]="dog_images/jpg/史毕诺犬"
    # label_list[23]="dog_images/jpg/吉娃娃"
    # label_list[24]="dog_images/jpg/哈士奇"
    # label_list[25]="dog_images/jpg/哈瓦那犬"
    # label_list[26]="dog_images/jpg/喜乐蒂牧羊犬"
    # label_list[27]="dog_images/jpg/圣伯纳犬"
    # label_list[28]="dog_images/jpg/墨西哥无毛犬"
    # label_list[29]="dog_images/jpg/大丹犬"
    # label_list[30]="dog_images/jpg/大瑞士山地犬"
    # label_list[31]="dog_images/jpg/大白熊犬"
    # label_list[32]="dog_images/jpg/大麦町犬"
    # label_list[33]="dog_images/jpg/奇努克犬"
    # label_list[34]="dog_images/jpg/威尔士梗"
    # label_list[35]="dog_images/jpg/威尔士跳猎犬"
    # label_list[36]="dog_images/jpg/威玛犬"
    # label_list[37]="dog_images/jpg/安纳托利亚牧羊犬"
    # label_list[38]="dog_images/jpg/寻血猎犬"
    # label_list[39]="dog_images/jpg/小型斗牛梗"
    # label_list[40]="dog_images/jpg/小型葡萄牙波登可犬"
    # label_list[41]="dog_images/jpg/小型雪纳瑞犬"
    # label_list[42]="dog_images/jpg/山地犬"
    # label_list[43]="dog_images/jpg/巨型雪纳瑞犬"
    # label_list[44]="dog_images/jpg/巴仙吉犬"
    # label_list[45]="dog_images/jpg/巴吉度犬"
    # label_list[46]="dog_images/jpg/巴哥犬"
    # label_list[47]="dog_images/jpg/布列塔尼犬"
    # label_list[48]="dog_images/jpg/布雷猎犬"
    # label_list[49]="dog_images/jpg/布鲁克浣熊猎犬"
    # label_list[50]="dog_images/jpg/布鲁塞尔格里芬犬"
    # label_list[51]="dog_images/jpg/帕尔森罗塞尔梗"
    # label_list[52]="dog_images/jpg/库瓦兹犬"
    # label_list[53]="dog_images/jpg/弗莱特寻回犬"
    # label_list[54]="dog_images/jpg/彭布罗克威尔士柯基犬"
    # label_list[55]="dog_images/jpg/德国宾莎犬"
    # label_list[56]="dog_images/jpg/德国牧羊犬"
    # label_list[57]="dog_images/jpg/德国短毛波音达"
    # label_list[58]="dog_images/jpg/德国硬毛波音达"
    # label_list[59]="dog_images/jpg/惠比特犬"
    # label_list[60]="dog_images/jpg/意大利灰狗"
    # label_list[61]="dog_images/jpg/戈登雪达犬"
    # label_list[62]="dog_images/jpg/拉布拉多寻回犬"
    # label_list[63]="dog_images/jpg/拉萨犬"
    # label_list[64]="dog_images/jpg/拳狮犬"
    # label_list[65]="dog_images/jpg/挪威伦德猎犬"
    # label_list[66]="dog_images/jpg/挪威布哈德犬"
    # label_list[67]="dog_images/jpg/挪威梗"
    # label_list[68]="dog_images/jpg/挪威猎鹿犬"
    # label_list[69]="dog_images/jpg/捕鼠梗"
    # label_list[70]="dog_images/jpg/捷克梗"
    # label_list[71]="dog_images/jpg/斗牛梗"
    # label_list[72]="dog_images/jpg/斗牛獒犬"
    # label_list[73]="dog_images/jpg/斯塔福郡斗牛梗"
    # label_list[74]="dog_images/jpg/新斯科舍猎鸭寻猎犬"
    # label_list[75]="dog_images/jpg/日本忡"
    # label_list[76]="dog_images/jpg/普罗特猎犬"
    # label_list[77]="dog_images/jpg/杜宾犬"
    # label_list[78]="dog_images/jpg/杰克罗素梗"
    # label_list[79]="dog_images/jpg/松狮"
    # label_list[80]="dog_images/jpg/柯利犬"
    # label_list[81]="dog_images/jpg/柴犬"
    # label_list[82]="dog_images/jpg/标准型雪纳瑞犬"
    # label_list[83]="dog_images/jpg/树丛浣熊猎犬"
    # label_list[84]="dog_images/jpg/格雷伊猎犬"
    # label_list[85]="dog_images/jpg/比利时牧羊犬"
    # label_list[86]="dog_images/jpg/比利时特伏丹犬"
    # label_list[87]="dog_images/jpg/比利时马林诺斯犬"
    # label_list[88]="dog_images/jpg/比利牛斯牧羊犬"
    # label_list[89]="dog_images/jpg/比格猎犬"
    # label_list[90]="dog_images/jpg/法国斗牛犬"
    # label_list[91]="dog_images/jpg/法国狼犬"
    # label_list[92]="dog_images/jpg/法老王猎犬"
    # label_list[93]="dog_images/jpg/波兰低地牧羊犬"
    # label_list[94]="dog_images/jpg/波兰德斯布比野犬"
    # label_list[95]="dog_images/jpg/波利犬"
    # label_list[96]="dog_images/jpg/波士顿梗"
    # label_list[97]="dog_images/jpg/波尔多犬"
    # label_list[98]="dog_images/jpg/波音达"
    # label_list[99]="dog_images/jpg/湖畔梗"
    # label_list[100]="dog_images/jpg/澳大利亚梗"
    # label_list[101]="dog_images/jpg/澳大利亚牧牛犬"
    # label_list[102]="dog_images/jpg/澳大利亚牧羊犬"
    # label_list[103]="dog_images/jpg/爱尔兰峡谷梗"
    # label_list[104]="dog_images/jpg/爱尔兰梗"
    # label_list[105]="dog_images/jpg/爱尔兰水猎犬"
    # label_list[106]="dog_images/jpg/爱尔兰猎狼犬"
    # label_list[107]="dog_images/jpg/爱尔兰红白雪达犬"
    # label_list[108]="dog_images/jpg/爱尔兰软毛梗"
    # label_list[109]="dog_images/jpg/猎兔犬"
    # label_list[110]="dog_images/jpg/猎水獭犬"
    # label_list[111]="dog_images/jpg/猴头梗"
    # label_list[112]="dog_images/jpg/玩具曼彻斯特犬"
    # label_list[113]="dog_images/jpg/玩具猎狐梗"
    # label_list[114]="dog_images/jpg/瑞典柯基犬"
    # label_list[115]="dog_images/jpg/田野小猎犬"
    # label_list[116]="dog_images/jpg/短毛猎狐梗"
    # label_list[117]="dog_images/jpg/短脚长身梗"
    # label_list[118]="dog_images/jpg/硬毛指示格里芬犬"
    # label_list[119]="dog_images/jpg/秋田犬"
    # label_list[120]="dog_images/jpg/粗毛柯利犬"
    # label_list[121]="dog_images/jpg/红骨猎浣熊犬"
    # label_list[122]="dog_images/jpg/约克夏梗"
    # label_list[123]="dog_images/jpg/纽芬兰犬"
    # label_list[124]="dog_images/jpg/维希拉猎犬"
    # label_list[125]="dog_images/jpg/罗威纳犬"
    # label_list[126]="dog_images/jpg/罗得西亚脊背犬"
    # label_list[127]="dog_images/jpg/罗秦犬"
    # label_list[128]="dog_images/jpg/美国可卡犬"
    # label_list[129]="dog_images/jpg/美国斯塔福郡梗"
    # label_list[130]="dog_images/jpg/美国水猎犬"
    # label_list[131]="dog_images/jpg/美国爱斯基摩犬"
    # label_list[132]="dog_images/jpg/美国猎狐犬"
    # label_list[133]="dog_images/jpg/美国英国猎浣熊犬"
    # label_list[134]="dog_images/jpg/腊肠犬"
    # label_list[135]="dog_images/jpg/芬兰拉普猎犬"
    # label_list[136]="dog_images/jpg/芬兰波美拉尼亚丝毛狗"
    # label_list[137]="dog_images/jpg/苏俄猎狼犬"
    # label_list[138]="dog_images/jpg/苏塞克斯猎犬"
    # label_list[139]="dog_images/jpg/苏格兰梗"
    # label_list[140]="dog_images/jpg/苏格兰猎鹿犬"
    # label_list[141]="dog_images/jpg/英国可卡犬"
    # label_list[142]="dog_images/jpg/英国斗牛犬"
    # label_list[143]="dog_images/jpg/英国猎狐犬"
    # label_list[144]="dog_images/jpg/英国玩具犬"
    # label_list[145]="dog_images/jpg/英国跳猎犬"
    # label_list[146]="dog_images/jpg/英格兰雪达犬"
    # label_list[147]="dog_images/jpg/荷兰毛狮犬"
    # label_list[148]="dog_images/jpg/萨摩耶犬"
    # label_list[149]="dog_images/jpg/萨路基猎犬"
    # label_list[150]="dog_images/jpg/葡萄牙水犬"
    # label_list[151]="dog_images/jpg/藏獒"
    # label_list[152]="dog_images/jpg/蝴蝶犬"
    # label_list[153]="dog_images/jpg/西帕基犬"
    # label_list[154]="dog_images/jpg/西施犬"
    # label_list[155]="dog_images/jpg/西班牙小猎犬"
    # label_list[156]="dog_images/jpg/西藏梗"
    # label_list[157]="dog_images/jpg/西藏猎犬"
    # label_list[158]="dog_images/jpg/西高地白梗"
    # label_list[159]="dog_images/jpg/诺福克梗"
    # label_list[160]="dog_images/jpg/贝灵顿梗"
    # label_list[161]="dog_images/jpg/贵宾犬"
    # label_list[162]="dog_images/jpg/边境牧羊犬"
    # label_list[163]="dog_images/jpg/迦南犬"
    # label_list[164]="dog_images/jpg/迷你杜宾"
    # label_list[165]="dog_images/jpg/迷你贝吉格里芬凡丁犬"
    # label_list[166]="dog_images/jpg/那不勒斯獒"
    # label_list[167]="dog_images/jpg/金毛寻回犬"
    # label_list[168]="dog_images/jpg/锡利哈姆梗"
    # label_list[169]="dog_images/jpg/阿富汗猎犬"
    # label_list[170]="dog_images/jpg/阿拉斯加雪橇犬"
    # label_list[171]="dog_images/jpg/马士提夫獒犬"
    # label_list[172]="dog_images/jpg/马尔济斯犬"
    # label_list[173]="dog_images/jpg/骑士查理王小猎犬"
    # label_list[174]="dog_images/jpg/黑俄罗斯梗"
    # label_list[175]="dog_images/jpg/黑褐猎浣熊犬"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='./',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
