# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import os
import scipy
import numpy as np
from tflearn.data_utils import *
from tensorflow.python.lib.io import file_io

class googlenet_dog_model:
    __instance = None
    model = None

    def __init__(self,model_path="model_dog.tfl"):
        network = input_data(shape=[None, 227, 227, 3])
        conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name='conv1_7_7_s2')
        pool1_3_3 = max_pool_2d(conv1_7_7, 3, strides=2)
        pool1_3_3 = local_response_normalization(pool1_3_3)
        conv2_3_3_reduce = conv_2d(pool1_3_3, 64, 1, activation='relu', name='conv2_3_3_reduce')
        conv2_3_3 = conv_2d(conv2_3_3_reduce, 192, 3, activation='relu', name='conv2_3_3')
        conv2_3_3 = local_response_normalization(conv2_3_3)
        pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
        inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
        inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96, 1, activation='relu', name='inception_3a_3_3_reduce')
        inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128, filter_size=3, activation='relu',
                                   name='inception_3a_3_3')
        inception_3a_5_5_reduce = conv_2d(pool2_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_3a_5_5_reduce')
        inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu',
                                   name='inception_3a_5_5')
        inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
        inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu',
                                        name='inception_3a_pool_1_1')

        # merge the inception_3a__
        inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1],
                                    mode='concat', axis=3)

        inception_3b_1_1 = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_1_1')
        inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu',
                                          name='inception_3b_3_3_reduce')
        inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3, activation='relu',
                                   name='inception_3b_3_3')
        inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu',
                                          name='inception_3b_5_5_reduce')
        inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5, name='inception_3b_5_5')
        inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1, name='inception_3b_pool')
        inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_3b_pool_1_1')

        # merge the inception_3b_*
        inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1],
                                    mode='concat', axis=3, name='inception_3b_output')

        pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
        inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu',
                                          name='inception_4a_3_3_reduce')
        inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3, activation='relu',
                                   name='inception_4a_3_3')
        inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu',
                                          name='inception_4a_5_5_reduce')
        inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5, activation='relu',
                                   name='inception_4a_5_5')
        inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1, name='inception_4a_pool')
        inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4a_pool_1_1')

        inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1],
                                    mode='concat', axis=3, name='inception_4a_output')

        inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
        inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu',
                                          name='inception_4b_3_3_reduce')
        inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu',
                                   name='inception_4b_3_3')
        inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu',
                                          name='inception_4b_5_5_reduce')
        inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4b_5_5')

        inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1, name='inception_4b_pool')
        inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4b_pool_1_1')

        inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1],
                                    mode='concat', axis=3, name='inception_4b_output')

        inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_1_1')
        inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',
                                          name='inception_4c_3_3_reduce')
        inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256, filter_size=3, activation='relu',
                                   name='inception_4c_3_3')
        inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu',
                                          name='inception_4c_5_5_reduce')
        inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4c_5_5')

        inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
        inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4c_pool_1_1')

        inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1],
                                    mode='concat', axis=3, name='inception_4c_output')

        inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
        inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu',
                                          name='inception_4d_3_3_reduce')
        inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu',
                                   name='inception_4d_3_3')
        inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu',
                                          name='inception_4d_5_5_reduce')
        inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5, activation='relu',
                                   name='inception_4d_5_5')
        inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1, name='inception_4d_pool')
        inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu',
                                        name='inception_4d_pool_1_1')

        inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1],
                                    mode='concat', axis=3, name='inception_4d_output')

        inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
        inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu',
                                          name='inception_4e_3_3_reduce')
        inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_4e_3_3')
        inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu',
                                          name='inception_4e_5_5_reduce')
        inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_4e_5_5')
        inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1, name='inception_4e_pool')
        inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu',
                                        name='inception_4e_pool_1_1')

        inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5, inception_4e_pool_1_1],
                                    axis=3, mode='concat')

        pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')

        inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
        inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu',
                                          name='inception_5a_3_3_reduce')
        inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu',
                                   name='inception_5a_3_3')
        inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu',
                                          name='inception_5a_5_5_reduce')
        inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5a_5_5')
        inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1, name='inception_5a_pool')
        inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5a_pool_1_1')

        inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1],
                                    axis=3, mode='concat')

        inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1, activation='relu', name='inception_5b_1_1')
        inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu',
                                          name='inception_5b_3_3_reduce')
        inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384, filter_size=3, activation='relu',
                                   name='inception_5b_3_3')
        inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu',
                                          name='inception_5b_5_5_reduce')
        inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce, 128, filter_size=5, activation='relu',
                                   name='inception_5b_5_5')
        inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1, name='inception_5b_pool')
        inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu',
                                        name='inception_5b_pool_1_1')
        inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1],
                                    axis=3, mode='concat')

        pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
        pool5_7_7 = dropout(pool5_7_7, 0.4)
        loss = fully_connected(pool5_7_7, 12, activation='softmax')
        network = regression(loss, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.001)
        self.model = tflearn.DNN(network, checkpoint_path='model_dog',
                            max_checkpoints=1, tensorboard_verbose=2)
        # model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
        #           show_metric=True, batch_size=64, snapshot_step=200,
        #           snapshot_epoch=False, run_id='googlenet_oxflowers17')
        model_path = os.path.join(model_path)
        print(model_path)
        self.model.load(model_path)

    def __new__(cls, *args, **kwd):
        if googlenet_dog_model.__instance is None:
            googlenet_dog_model.__instance = object.__new__(cls, *args, **kwd)
        return googlenet_dog_model.__instance

    # 此处需数据集labels
    def get_labels(self):
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
        return label_list

    def run_inference_on_image(self,file_path):

        # img = scipy.ndimage.imread(file_path, mode="RGB")
        # img = scipy.misc.imresize(img, (227, 227), interp="bicubic").astype(np.float32, casting='unsafe')
        im = load_image(file_path)
        im = resize_image(im, 227, 227)
        im = pil_to_nparray(im)
        im /= 255.

        # Predict
        prediction = self.model.predict([im])
        predict_label = self.model.predict_label([im])
        print(prediction)
        print(predict_label)
        print(max(prediction[0]))
        print(prediction[0].tolist().index(max(prediction[0])))
        label_list = self.get_labels()

        result = ""
        result_flag = False
        result_index = None
        max_index = prediction[0].tolist().index(max(prediction[0]))
        for i in range(len(prediction[0])):
            if prediction[0][i] > 0.8 and i==max_index:
                result_flag = True
            else:
                if prediction[0][i] > 0.05:
                    if result=="":
                        result = "也可能是：<br/>"
                    result += "<li>" + label_list[i] + " 可能性：" + str(prediction[0][i]) + "</li>" + "<br/>"

        if result_flag:
            result = "这是 <b>%s</b>" % (label_list[prediction[0].tolist().index(max(prediction[0]))]) + "<br/>" + result
        else:
            result = "sorry, 没有确切答案" + "<br/>" + result
        print(result_flag)
        print(result)
        return result
        #
        #
        # print(file_path)
        # img = scipy.ndimage.imread(file_path, mode="RGB")
        # print("okok")
        # img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
        #
        # # Predict
        # prediction = self.model.predict([img])
        # predict_label = self.model.predict_label([img])
        # print(prediction)
        # print(predict_label)
        # print(prediction)
        # print(prediction[0].tolist().index(max(prediction[0])))
        # num = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # return ("This is a %s" % (num[prediction[0].tolist().index(max(prediction[0]))]))