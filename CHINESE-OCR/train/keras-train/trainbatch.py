# -*- coding: utf-8 -*-
import dataset
import keys_keras
import numpy as np
import torch
import time
import os
import sys
sys.path.insert(0, os.getcwd())
import tensorflow as tf
import pydot
import graphviz
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
from keras.utils import plot_model

characters = keys_keras.alphabet[:]
# from model import get_model
from keras.layers import Flatten, BatchNormalization, Permute, TimeDistributed, Dense, Bidirectional, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
rnnunit = 256
from keras import backend as K

from keras.layers import Lambda
from keras.optimizers import SGD


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # print("cccccccccc:",y_pred,labels,input_length,label_length)
    y_pred = y_pred[:, 2:, :]

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(height, nclass, learning_rate):
    input = Input(shape=(height, None, 1), name='the_input')
    m = Conv2D(64, kernel_size=(3, 3), activation='relu',
               padding='same', name='conv1')(input)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(m)
    m = Conv2D(128, kernel_size=(3, 3), activation='relu',
               padding='same', name='conv2')(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu',
               padding='same', name='conv3')(m)
    m = Conv2D(256, kernel_size=(3, 3), activation='relu',
               padding='same', name='conv4')(m)

    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 1), padding='valid', name='pool3')(m)

    m = Conv2D(512, kernel_size=(3, 3), activation='relu',
               padding='same', name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512, kernel_size=(3, 3), activation='relu',
               padding='same', name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0, 1))(m)
    m = MaxPooling2D(pool_size=(2, 2), strides=(
        2, 1), padding='valid', name='pool4')(m)
    m = Conv2D(512, kernel_size=(2, 2), activation='relu',
               padding='valid', name='conv7')(m)

    # Permute层将输入的维度按照给定模式进行重排，例如，当需要将RNN和CNN网络连接时，可能会用到该层。
    m = Permute((2, 1, 3), name='permute')(m)
    m = TimeDistributed(Flatten(), name='timedistrib')(m)
    # cnn之后链接双向GRU，双向GRU会输出固定长度的序列，这是一个encode的过程，之后再连接一个双向GRU，对该序列进行解码
    # 该序列的输出为长度为256的序列
    # cnn之后连接双向GRU
    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm1')(m)
    # 全连接层-rnnunit为全连接层的输出维度
    m = Dense(rnnunit, name='blstm1_out', activation='linear')(m)
    # 连接双向GRU
    m = Bidirectional(GRU(rnnunit, return_sequences=True), name='blstm2')(m)
    # 全连接输出
    y_pred = Dense(nclass, name='blstm2_out', activation='softmax')(m)
    # 确定模型
    basemodel = Model(inputs=input, outputs=y_pred)

    labels = Input(name='the_labels', shape=[None, ], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length,
                          label_length], outputs=[loss_out])
    sgd = SGD(lr=learning_rate, decay=1e-6,
              momentum=0.9, nesterov=True, clipnorm=5)
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.summary()
    return model, basemodel


nclass = len(characters) + 1
trainroot = '/home/gu/media/crnn_data/mytrain'
valroot = '/home/gu/media/crnn_data/myval'
# modelPath = '../pretrain-models/keras.hdf5'
# modelPath = '/home/gu/workspace/CHINESE-OCR/checkpoints/my_model_keras.h5'
modelPath = '/home/gu/workspace/CHINESE-OCR/mymodels/my_model_keras.h5'
workers = 4
imgH = 32
imgW = 256
keep_ratio = False
random_sample = False
batchSize = 32
testSize = 16
n_len = 50
loss = 1000
interval = 50
LEARNING_RATE = 0.01
Learning_decay_step = 20000
PERCEPTION = 0.3
EPOCH_NUMS = 1000000
MODEL_PATH = '/home/gu/workspace/CHINESE-OCR/mymodels/'
MODEL_NAME = '/my_model_keras_v2.h5'
LOG_FILE = 'log.txt'
SUMMARY_PATH = './log/'
if not os.path.exists(MODEL_PATH):
    print('Creating save model path!!')
    os.makedirs(MODEL_PATH)
if not os.path.exists(SUMMARY_PATH):
    os.makedirs(SUMMARY_PATH)

model, basemodel = get_model(
    height=imgH, nclass=nclass, learning_rate=LEARNING_RATE)

config = tf.ConfigProto(intra_op_parallelism_threads=2)
config.gpu_options.per_process_gpu_memory_fraction = PERCEPTION
KTF.set_session(tf.Session(config=config))

# 加载预训练参数
if os.path.exists(modelPath):
    # basemodel.load_weights(modelPath)
    model.load_weights(modelPath)

plot_model(basemodel, to_file='basemodel.png')
plot_model(model, to_file='model.png')


def one_hot(text, length=10, characters=characters):
    label = np.zeros(length)
    for i, char in enumerate(text):
        index = characters.find(char)
        if index == -1:
            index = characters.find(u' ')
        if i < length:
            label[i] = index
    return label


# 导入数据
if random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, batchSize)
else:
    sampler = None
train_dataset = dataset.lmdbDataset(root=trainroot, target_transform=one_hot)
# print(len(train_dataset))

test_dataset = dataset.lmdbDataset(
    root=valroot,
    transform=dataset.resizeNormalize((imgW, imgH)),
    target_transform=one_hot)

# 生成训练用数据
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchSize,
    shuffle=True,
    sampler=sampler,
    num_workers=int(workers),
    collate_fn=dataset.alignCollate(
        imgH=imgH, imgW=imgW, keep_ratio=keep_ratio))

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=testSize, shuffle=True, num_workers=int(workers))

j = 0
print('Strat training!!')
for i in range(EPOCH_NUMS):
    for X, Y in train_loader:
        start = time.time()
        X = X.numpy()
        X = X.reshape((-1, imgH, imgW, 1))
        Y = np.array(Y)
        Length = int(imgW / 4) - 2
        batch = X.shape[0]
        X_train, Y_train = [X, Y,
                            np.ones(batch) * Length,
                            np.ones(batch) * n_len], np.ones(batch)
        print('IMG_SHAPE:', np.shape(X))
        print('LABEL_SHAPE:', np.shape(Y))
        # print(np.shape(X_train))
        model.train_on_batch(X_train, Y_train)
        if j % interval == 0:
            times = time.time() - start
            currentLoss_train = model.evaluate(X_train, Y_train)
            X, Y = next(iter(test_loader))
            X = X.numpy()
            X = X.reshape((-1, imgH, imgW, 1))
            Y = Y.numpy()
            Y = np.array(Y)
            batch = X.shape[0]
            X_val, Y_val = [
                X, Y, np.ones(batch) * Length,
                np.ones(batch) * n_len], np.ones(batch)
            crrentLoss = model.evaluate(X_val, Y_val)
            print('Learning rate is: ', LEARNING_RATE)
            now_time = time.strftime('%Y/%m/%d-%H:%M:%S',
                                     time.localtime(time.time()))
            print('Time: [%s]--Step/Epoch/Total: [%d/%d/%d]' % (now_time, j, i,
                                                                EPOCH_NUMS))
            print('\tTraining Loss is: [{}]'.format(currentLoss_train))
            print('\tVal Loss is: [{}]'.format(crrentLoss))
            print('\tSpeed is: [{}] Samples/Secs'.format(interval / times))
            path = MODEL_PATH + MODEL_NAME
            with open(LOG_FILE, mode='a') as log_file:
                log_str = now_time + '----global_step:' + str(
                    j) + '----loss:' + str(loss) + '\n'
                log_file.writelines(log_str)
            log_file.close()
            print('\tWriting to the file: log.txt')
            print("\tSave model to disk: {}".format(path))
            model.save(path)
            if crrentLoss < loss:
                loss = crrentLoss
        if j > 0 and j % Learning_decay_step == 0:
            LEARNING_RATE_ori = LEARNING_RATE
            LEARNING_RATE = 0.5 * LEARNING_RATE
            print('\tUpdating Leaning rate from {} to {}'.format(
                LEARNING_RATE_ori, LEARNING_RATE))
        j += 1
