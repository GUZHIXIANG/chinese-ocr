import cv2
import numpy as np
import os
from collections import defaultdict
from tools import AvoidZero
from tools_new import ConvertData
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import itertools

from DefData import Line, Point, Rect


class PostProcessing():
    '''
    ########################################  Introduction  ########################################
    This is the class of post processing after edge has been detected by the HED network

    ######################################## data structure #########################################
    Point: x,y
    Line: pt1,pt2,k,b

    ########################################  Variables  ############################################
    img_org: Original image
    img_edges: Output of HED network, containing 6 edge images. Usually we use img_edges[5]
    img_edges_bin: Binarized edge images by opencv function threshold. Default threshold 120
    '''

    def __init__(self, _img_org, _img_edges, bin_th=120, _Debug=False):
        self.img_org = _img_org
        self.img_edges = _img_edges
        self.h, self.w = _img_edges.shape
        self.img_size = self.w * self.h

        bin_th = 120
        self.Debug = _Debug
        self.img_edges_bin = cv2.threshold(
            _img_edges, bin_th, 255, cv2.THRESH_BINARY_INV)[1]
        self.lines = cv2.HoughLinesP(self.img_edges_bin, 0.8, np.pi / 180, 90,
                                     minLineLength=50, maxLineGap=10)
        self.lines = ConvertData(self.lines)
        self.fullines = self.GetFullLines(self.lines)

        if self.Debug:
            self.img_flines = self.img_org.copy()
            self.img_final = self.img_org.copy()
            self.debug_path = './Debug'
            if not os.path.exists(self.debug_path):
                os.makedirs(self.debug_path)
            cv2.imwrite(os.path.join(self.debug_path,
                                     'img_original.jpg'), self.img_org)
            cv2.imwrite(os.path.join(self.debug_path,
                                     'edge_original.jpg'), self.img_edges)
            cv2.imwrite(os.path.join(self.debug_path,
                                     'edge_binary.jpg'), self.img_edges_bin)
            img_houghlines = self.img_org.copy()
            for line in self.lines:
                x1, y1, x2, y2 = line.pt1.x, line.pt1.y, line.pt2.x, line.pt2.y
                cv2.line(img_houghlines, (x1, y1), (x2, y2),
                         (200, 100, 0), 1, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.debug_path,
                                     'img_houghlines.jpg'), img_houghlines)

            #f = open(os.path.join(self.debug_path, 'Hough_Points.txt'), 'w')
            # for line in self.lines:

            # k = line.k
            # b = line.b
            # pt1 = Point(0, int(b))
            # pt2 = Point(int(-b / AvoidZero(k)), 0)
            # pt3 = Point(self.w, int(k * self.w + b))
            # pt4 = Point(int((self.h - b) / AvoidZero(k)), self.h)
            # #print(pt1,pt2,pt3,pt4)
            # tmp = []
            # for pt in [pt1, pt2, pt3, pt4]:
            #     if pt.x >= 0 and pt.x <= self.w and pt.y >= 0 and pt.y <= self.h:
            #         tmp.append(pt)
            # if len(tmp) == 2:
            #     l = Line(tmp[0], tmp[1])
            #info = '{},{},{},{}\n'.format(int(l.pt1.x), int(l.pt1.y), int(l.pt2.x), int(l.pt2.y))
            # f.write(info)
            # f.close()

    def GetFullLines(self, _lines):
        lines = []
        for line in _lines:
            k = line.k
            b = line.b
            pt1 = Point(0, int(b))
            pt2 = Point(int(-b / AvoidZero(k)), 0)
            pt3 = Point(self.w, int(k * self.w + b))
            pt4 = Point(int((self.h - b) / AvoidZero(k)), self.h)
            # print(pt1,pt2,pt3,pt4)
            tmp = []
            for pt in [pt1, pt2, pt3, pt4]:
                if pt.x >= 0 and pt.x <= self.w and pt.y >= 0 and pt.y <= self.h:
                    tmp.append(pt)
            if len(tmp) == 2:
                l = Line(tmp[0], tmp[1])
                lines.append(l)
        return np.array(lines)

    def ClusterLines_Kmeans(self, lines):
        points = []
        for line in lines:
            pt1 = line.pt1
            pt2 = line.pt2
            points.append([pt1.x, pt1.y, pt2.x, pt2.y])
        points = np.array(points)
        # print(points[:10])
        X = np.zeros((2 * len(points), 2))
        X[:len(points)] = points[:, :2]
        X[len(points):] = points[:, 2:]
        sc_scores = dict()
        clusters = range(10, 21, 2)
        models = dict()
        for k in clusters:
            model = KMeans(n_clusters=k, random_state=0).fit(X)
            sc_score = silhouette_score(X, model.labels_, metric='euclidean')
            sc_scores[k] = sc_score
            models[k] = model
        max_sc = max(sc_scores, key=sc_scores.get)
        # print(max_sc)
        labels = models[max_sc].labels_
        classed_lines = defaultdict(list)
        for i, x in enumerate(zip(labels[:len(points)], labels[len(points):])):
            set_x = set(x)
            if len(set_x) == 2:
                classed_lines[str(set_x)].append(i)
        line_set = list(classed_lines.values())
        return line_set

    def GetMKB(self, _line_set):
        k_all = []
        b_all = []
        for line in _line_set:
            k_all.append(line.k)
            b_all.append(line.b)
        return np.median(k_all), np.median(b_all)

    def GetCrossPoint(self, _lines):
        pts = []
        for line_pair in itertools.combinations(_lines, 2):
            lineA = line_pair[0]
            lineB = line_pair[1]
            x = (lineA.k * lineA.pt1.x - lineA.pt1.y - lineB.b *
                 lineB.pt1.x + lineB.pt1.y) / AvoidZero(lineA.k - lineB.k)
            y = (lineA.k * lineB.k * (lineA.pt1.x - lineB.pt1.x) + lineA.k *
                 lineB.pt1.y - lineB.k * lineA.pt1.y) / AvoidZero(lineA.k - lineB.k)
            if x >= 0 and x <= self.w and y >= 0 and y <= self.h:
                pts.append(Point(int(x), int(y)))
                if self.Debug:
                    cv2.circle(self.img_flines, (int(x), int(y)),
                               3, (0, 0, 255), -1)
                    cv2.imwrite(os.path.join(self.debug_path,
                                             'img_flines_crosspoints.jpg'), self.img_flines)
        return pts

    def GetRectFromCrosspts(self, _crosspoints):
        rects = []
        for cand_pts in itertools.combinations(_crosspoints, 4):
            #x_mean = np.array([cand_pts[0].x, cand_pts[1].x, cand_pts[2].x, cand_pts[3].x]).mean
            #y_mean = np.array([cand_pts[0].y, cand_pts[1].y, cand_pts[2].y, cand_pts[3].y]).mean
            rect = Rect(cand_pts[0], cand_pts[1], cand_pts[2], cand_pts[3])
            if rect.isRect:
                if (rect.ratio > 0.1 or rect.ratio < 10) and rect.area / self.img_size > 0.25:
                    rects.append(rect)
        return rects

    def GetMergedLines(self, _merged_lines):
        flines_kb = []
        for lines in _merged_lines:
            line_set = []
            for index in lines:
                line_set.append(self.fullines[index])
            k, b = self.GetMKB(line_set)
            flines_kb.append([k, b])

        flines = []
        for line_kb in flines_kb:
            # print(line_kb)
            k = line_kb[0]
            b = line_kb[1]
            pt1 = Point(0, int(b))
            pt2 = Point(int(-b / AvoidZero(k)), 0)
            pt3 = Point(self.w, int(k * self.w + b))
            pt4 = Point(int((self.h - b) / AvoidZero(k)), self.h)
            # print(pt1,pt2,pt3,pt4)
            tmp = []
            for pt in [pt1, pt2, pt3, pt4]:
                if pt.x >= 0 and pt.x <= self.w and pt.y >= 0 and pt.y <= self.h:
                    tmp.append(pt)
            if len(tmp) == 2:
                l = Line(tmp[0], tmp[1])
                if self.Debug:
                    color = (random.randint(0, 255), random.randint(
                        0, 255), random.randint(0, 255))
                    cv2.line(self.img_flines, (l.pt1.x, l.pt1.y),
                             (l.pt2.x, l.pt2.y), color, 2, lineType=cv2.LINE_AA)
                flines.append(l)
        flines.append(Line(Point(0, 0), Point(self.w, 0)))
        flines.append(Line(Point(0, 0), Point(0, self.h)))
        flines.append(Line(Point(self.w, 0), Point(self.w, self.h)))
        flines.append(Line(Point(0, self.h), Point(self.w, self.h)))
        if self.Debug:
            cv2.imwrite(os.path.join(self.debug_path,
                                     'img_flines.jpg'), self.img_flines)
        return flines

    def GetLineEnergy(self, _line):
        xmin = np.min(np.array([_line.pt1.x - 1, _line.pt2.x - 1]))
        ymin = np.min(np.array([_line.pt1.y - 1, _line.pt2.y - 1]))
        xmax = np.max(np.array([_line.pt1.x + 1, _line.pt2.x + 1]))
        ymax = np.max(np.array([_line.pt1.y + 1, _line.pt2.y + 1]))
        # print('###########')
        # print(xmin,ymin,xmax,ymax)
        line_area = self.img_edges_bin[ymin:ymax, xmin:xmax]
        # cv2.imshow('la',line_area)
        # cv2.waitKey(0)
        energy = np.sum(line_area)
        return float(energy)

    def GetRectEnergy(self, _rect):
        total = self.GetLineEnergy(_rect.side_up) + self.GetLineEnergy(_rect.side_bottom) + \
            self.GetLineEnergy(_rect.side_left) + \
            self.GetLineEnergy(_rect.side_right)
        return total

    def GetDocRect(self, _rects):
        doc = False
        eng = 0
        for rect in _rects:
            e = self.GetRectEnergy(rect)
            if e > eng:
                eng = e
                doc = rect
        return doc

    def process(self):
        # print(len(self.fullines))
        MergedLines = self.ClusterLines_Kmeans(self.fullines)
        # flines_kb = []
        # for lines in MergedLines:
        #     line_set = []
        #     for index in lines:
        #         line_set.append(self.houghlines[index])
        #     k, b = self.MergeLines(line_set)
        #     flines_kb.append([k, b])

        flines = self.GetMergedLines(MergedLines)
        print(len(flines))

        # flines = self.ClusterLines_Kmeans(flines)
        # flines = self.GetMergedLines(flines)
        # img_f = self.img_org.copy()
        # for l in flines:
        #     color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #     cv2.line(img_f, (l.pt1.x, l.pt1.y), (l.pt2.x, l.pt2.y), color, 2, lineType=cv2.LINE_AA)
        #     cv2.imwrite(os.path.join(self.debug_path, 'img_flines_2nd.jpg'), img_f)

        crosspoints = self.GetCrossPoint(flines)

        # print(len(crosspoints))

        cand_rects = self.GetRectFromCrosspts(crosspoints)

        # print(cand_rects)
        document = self.GetDocRect(cand_rects)
        # print(document)
        if self.Debug:
            cv2.line(self.img_final, (document.pt_ul.x, document.pt_ul.y),
                     (document.pt_ur.x, document.pt_ur.y), (200, 100, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(self.img_final, (document.pt_ur.x, document.pt_ur.y),
                     (document.pt_br.x, document.pt_br.y), (200, 100, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(self.img_final, (document.pt_ul.x, document.pt_ul.y),
                     (document.pt_bl.x, document.pt_bl.y), (200, 100, 0), 3, lineType=cv2.LINE_AA)
            cv2.line(self.img_final, (document.pt_br.x, document.pt_br.y),
                     (document.pt_bl.x, document.pt_bl.y), (200, 100, 0), 3, lineType=cv2.LINE_AA)
            cv2.imwrite(os.path.join(self.debug_path,
                                     'img_final.jpg'), self.img_final)
        # print(self.w, self.h)
        # for line in flines_kb:
        #     k = line[0]
        #     b = line[1]
        #     pt1 = Point(0, int(b))
        #     pt2 = Point(int(-b / AvoidZero(k)), 0)
        #     pt3 = Point(self.w, int(k * self.w + b))
        #     pt4 = Point(int((self.h - b) / AvoidZero(k)), self.h)
        #     print(pt1,pt2,pt3,pt4)
        #     tmp = []
        #     for pt in [pt1, pt2, pt3, pt4]:
        #         if pt.x >= 0 and pt.x <= self.w and pt.y >= 0 and pt.y <= self.h:
        #             tmp.append(pt)
        #     if len(tmp) == 2:
        #         l = Line(tmp[0], tmp[1])
        #         if self.Debug:
        #             color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #             cv2.line(self.img_flines, (l.pt1.x, l.pt1.y), (l.pt2.x, l.pt2.y), color, 2, lineType=cv2.LINE_AA)
        #         flines.append(l)
        # print(flines)
        # if self.Debug:
        #     cv2.imwrite(os.path.join(self.debug_path, 'img_flines.jpg'), self.img_flines)
