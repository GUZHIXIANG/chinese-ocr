# coding:utf-8
import time
from glob import glob

import numpy as np
from PIL import Image

import model
# ces

paths = glob('./test/*.*')

if __name__ == '__main__':

    testPath = 'CHINESE-OCR/test/'

    with open(testPath+'labels.txt', 'r') as file:
        labelfile = [x for x in file]

    with open(testPath+'res100_v2.txt', 'w') as file:
        root = '00000000'
        for i in range(100):
            len_i = len(str(i))
            root = root[:-len_i] + str(i)
            file.write('target=' + root + '.jpg\n')
            file.write('real=\n'+labelfile[i])

            im = Image.open(testPath+"%s.jpg" % root)
            img = np.array(im.convert('RGB'))

            '''
            result,img,angel分别对应-识别结果，图像的数组，文字旋转角度
            '''
            result, img, angle = model.model(
                img, model='keras', adjust=True, detectAngle=False)

            file.write('pred=\n')
            for key in result:
                file.write(result[key][1] + '\n')

            file.write("---------------------------------------\n")
