import numpy as np
from tools import AvoidZero
from collections import defaultdict


def dist(pt1, pt2):
    return np.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)


class Point:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y


class Line:
    def __init__(self, _pt1, _pt2):
        self.pt1 = _pt1
        self.pt2 = _pt2
        self.k = float(_pt1.y - _pt2.y) / AvoidZero(_pt1.x - _pt2.x)
        self.b = _pt1.y - self.k * _pt1.x
        self.length = dist(_pt1, _pt2)


class Rect:
    def __init__(self, _pt1, _pt2, _pt3, _pt4):
        self.pt1 = _pt1
        self.pt2 = _pt2
        self.pt3 = _pt3
        self.pt4 = _pt4
        self.pt_ul = None
        self.pt_ur = None
        self.pt_bl = None
        self.pt_br = None
        self.side_up = None
        self.side_bottom = None
        self.side_left = None
        self.side_right = None
        self.ratio = None
        self.area = None
        self.pt_array = np.array([[_pt1.x, _pt1.y], [_pt2.x, _pt2.y], [
                                 _pt3.x, _pt3.y], [_pt4.x, _pt4.y]])
        self.isRect = self.ifRect()
        #self.isConvex = False

    def ifRect(self):
        #x_mean = np.array([self.pt1.x, self.pt2.x, self.pt3.x, self.pt4.x]).mean
        #y_mean = np.array([self.pt1.y, self.pt2.y, self.pt3.y, self.pt4.y]).mean
        mean = self.pt_array.mean(axis=0)
        tmp = defaultdict(list)
        for pt in self.pt_array:
            d_pt = pt - mean
            if d_pt[0] > 0:
                if d_pt[1] > 0:
                    tmp[2].append(pt)
                else:
                    tmp[3].append(pt)
            else:
                if d_pt[1] > 0:
                    tmp[1].append(pt)
                else:
                    tmp[4].append(pt)
        if len(tmp) != 4:
            return False
        else:
            tmp2 = list()
            for k in range(1, 5):
                tmp2.extend(tmp[k])
            tmp2 = np.array(tmp2)

            d12 = tmp2[0] - tmp2[1]
            d23 = tmp2[1] - tmp2[2]
            d34 = tmp2[2] - tmp2[3]
            d41 = tmp2[3] - tmp2[0]

            t1 = d41[0] * -d12[1] - d41[1] * -d12[0]
            t2 = d12[0] * -d23[1] - d12[1] * -d23[0]
            t3 = d23[0] * -d34[1] - d23[1] * -d34[0]
            t4 = d34[0] * -d41[1] - d34[1] * -d41[0]

            if t1 * t2 * t3 * t4 < 0:
                return False
            else:
                self.pt_ul = Point(tmp[4][0][0], tmp[4][0][1])
                self.pt_ur = Point(tmp[3][0][0], tmp[3][0][1])
                self.pt_bl = Point(tmp[1][0][0], tmp[1][0][1])
                # left-up, right-up, left-bottom, right-bottom
                self.pt_br = Point(tmp[2][0][0], tmp[2][0][1])
                self.side_up = Line(self.pt_ul, self.pt_ur)
                self.side_bottom = Line(self.pt_bl, self.pt_br)
                self.side_left = Line(self.pt_ul, self.pt_bl)
                self.side_right = Line(self.pt_ur, self.pt_br)
                self.ratio = self.side_up.length / self.side_left.length
                self.area = self.side_up.length * self.side_left.length
                #self.isRect = True
                return True
