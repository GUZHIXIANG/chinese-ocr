import cv2
import numpy as np
import math
import random
import itertools
from collections import defaultdict
from DefData import Point,Line
from PostProcessing import PostProcessing

img_path = './test_imgs/2.jpeg'
edgeimg_path = './test_output/2.jpeg/edge.jpg'#'./output/out_6.jpg'
#edgeimg_path = './output/out_6.jpg'

img_org = cv2.imread(img_path)
img_edge = cv2.imread(edgeimg_path, 0)

a = PostProcessing(img_org,img_edge, _Debug=True)

a.process()