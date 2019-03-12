# coding:utf-8
# 添加文本方向 检测模型，自动检测文字方向，0、90、180、270
from math import *

import cv2
import numpy as np
from PIL import Image
import sys

sys.path.append("ocr")
from angle.predict import predict as angle_detect  # 文字方向检测

from crnn.crnn import crnnOcr

from ctpn.text_detect import text_detect
from ocr.model import predict as ocr


def crnnRec(im, ocrMode='keras'):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    index = 0
    results = {}
    results[index] = list()
    image = Image.fromarray(im).convert('L')
    # 进行识别出的文字识别
    if ocrMode == 'keras':
        sim_pred = ocr(image)
    else:
        sim_pred = crnnOcr(image)

    results[index].append(sim_pred)  # 识别文字

    return results


def model(img, model='keras'):
    """
    @@param:img,
    @@param:model,选择的ocr模型，支持keras\pytorch版本
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向

    """

    result = crnnRec(img, model)
    return result
