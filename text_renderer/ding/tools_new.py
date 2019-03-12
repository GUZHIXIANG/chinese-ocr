import math
import itertools
import numpy as np
import cv2
from collections import defaultdict

from DefData import Point,Line

def AvoidZero(x):
    if x == 0:
        return 0.00000000000000000000000000001
    else:
        return float(x)

def ConvertData(lines):
    STD_lines = []
    if not len(lines) == 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pt1 = Point(x1, y1)
            pt2 = Point(x2, y2)
            line_std = Line(pt1,pt2)
            STD_lines.append(line_std)
        return np.array(STD_lines)
    else:
        return None


def GetCrossPoint(lineA, lineB):
    x = (lineA.k * lineA.pt1.x - lineA.pt1.y - lineB.b * lineB.pt1.x + lineB.pt1.y) / AvoidZero(lineA.k - lineB.k)
    y = (lineA.k * lineB.k * (lineA.pt1.x - lineB.pt1.x) + lineA.k * lineB.pt1.y - lineB.k * lineA.pt1.y) / AvoidZero(lineA.k - lineB.k)
    return Point(x,y)