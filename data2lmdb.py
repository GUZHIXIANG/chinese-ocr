# -*- coding: utf-8 -*-
# @Author: gu
# @Date:   2019-03-12 11:10:50
# @Last Modified by:   gu
# @Last Modified time: 2019-03-12 11:16:31

import glob
import os
import lmdb
import cv2
import numpy as np
import codecs


def readData(inputPath):
    labelPath = inputPath + 'tmp_labels.txt'

    imgLabelList = list()
    with open(labelPath, 'r') as file:
        for _, line in enumerate(file):
            textName, label = line.strip()[:8], line.strip()[9:]
            imagePath = inputPath + textName + '.jpg'
            imgLabelList.append((imagePath, label))
    imgLabelList = sorted(imgLabelList, key=lambda x: len(x[1]))

    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]
    return imgPaths, txtLists


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print('...................')
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i]).encode()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    inputPath = '/home/gu/media/crnn_data/mytrain/default/'
    outputPath = '/home/gu/media/crnn_data/mytrain/'
    imgPaths, txtLists = readData(inputPath)
    createDataset(outputPath, imgPaths, txtLists,
                  lexiconList=None, checkValid=True)
