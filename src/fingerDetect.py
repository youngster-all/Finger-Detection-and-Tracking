import cv2
import numpy as np
# import matplotlib.pyplot as plt


class Detector(object):

    def __init__(self, img):
        self.hand_rect_one_x=0
        self.hand_rect_two_x=0
        self.hand_rect_one_y=0
        self.hand_rect_two_y=0
        self.img=img
        self.traverse_point=[]

    def detectByColor(self):
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb) # 把图像转换到YUV色域
        (y, cr, cb) = cv2.split(ycrcb) # 图像分割, 分别获取y, cr, br通道图像
        # 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
        cr1 = cv2.GaussianBlur(cr, (5, 5), 0) # 对cr通道分量进行高斯滤波
        # 根据OTSU算法求图像阈值, 对图像进行二值化
        _, skin1 = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite('bw_img.png',skin1)
        return skin1

    def segmentByColor(self, mask):
        thresh = cv2.merge((mask,mask,mask))
        return cv2.bitwise_and(self.img, thresh)

    def draw_rect(self):
        rows, cols, _ = self.img.shape
        self.hand_rect_one_x = int(6*rows/20)
        self.hand_rect_one_y = int(9*cols/20)
        self.hand_rect_two_x = self.hand_rect_one_x + 30
        self.hand_rect_two_y = self.hand_rect_one_y + 30
        cv2.rectangle(self.img, (self.hand_rect_one_y, self.hand_rect_one_x),
                      (self.hand_rect_two_y, self.hand_rect_two_x), (0, 255, 0), 1)

        return self.img

    def hand_histogram(self):
        hsv_frame = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # roi = np.zeros([30,30, 3], dtype=hsv_frame.dtype)

        roi= hsv_frame[self.hand_rect_one_x:self.hand_rect_one_x +180,
            self.hand_rect_one_y:self.hand_rect_one_y +180 ]

        hand_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
        return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

    def contours(self, hist_mask_image):
        gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
        # cv2.imshow("002",thresh)
        _, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return cont

    def max_contour(self,contour_list):
        max_i = 0
        max_area = 0

        for i in range(len(contour_list)):
            cnt = contour_list[i]

            area_cnt = cv2.contourArea(cnt)

            if area_cnt > max_area:
                max_area = area_cnt
                max_i = i

        return contour_list[max_i]

    def centroid(self,max_contour):
        moment = cv2.moments(max_contour)
        if moment['m00'] != 0:
            cx = int(moment['m10'] / moment['m00'])
            cy = int(moment['m01'] / moment['m00'])
            return cx, cy
        else:
            return None

    def farthest_point(self,defects, contour, centroid):
        if defects is not None and centroid is not None:
            s = defects[:, 0][:, 0]
            cx, cy = centroid

            x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
            y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

            xp = cv2.pow(cv2.subtract(x, cx), 2)
            yp = cv2.pow(cv2.subtract(y, cy), 2)
            dist = cv2.sqrt(cv2.add(xp, yp))

            dist_max_i = np.argmax(dist)

            if dist_max_i < len(s):
                farthest_defect = s[dist_max_i]
                farthest_point = tuple(contour[farthest_defect][0])
                return farthest_point
            else:
                return None

    def hist_masking(self, hist):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], hist, [0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cv2.filter2D(dst, -1, disc, dst)

        ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

        # thresh = cv2.dilate(thresh, None, iterations=5)

        thresh = cv2.merge((thresh, thresh, thresh))

        return cv2.bitwise_and(self.img, thresh)

    def draw_circles(self,traverse_point):
        if traverse_point is not None:
            for i in range(len(traverse_point)):
                cv2.circle(self.img, traverse_point[i], int(5 - (5 * i * 3) / 100), [0, 255, 255], -1)

    def manage_image_opr(self,hand_hist):
        hist_mask_image = self.hist_masking(hand_hist)
        cv2.imshow("heap", hist_mask_image)
        contour_list = self.contours(hist_mask_image)
        max_cont =self.max_contour(contour_list)

        cnt_centroid = self.centroid(max_cont)
        cv2.circle(self.img, cnt_centroid, 5, [255, 0, 255], -1)

        if max_cont is not None:
            hull = cv2.convexHull(max_cont, returnPoints=False)
            defects = cv2.convexityDefects(max_cont, hull)
            far_point = self.farthest_point(defects, max_cont, cnt_centroid)
            print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
            cv2.circle(self.img, far_point, 5, [0, 0, 255], -1)
            if len(self.traverse_point) < 20:
                self.traverse_point.append(far_point)
            else:
                self.traverse_point.pop(0)
                self.traverse_point.append(far_point)

            self.draw_circles(self.traverse_point)

    def manage_image_opr_color(self,hand_mask_image):
        cv2.imshow("heap", hand_mask_image)
        contour_list = self.contours(hand_mask_image)
        max_cont =self.max_contour(contour_list)

        cnt_centroid = self.centroid(max_cont)
        cv2.circle(self.img, cnt_centroid, 5, [255, 0, 255], -1)

        if max_cont is not None:
            hull = cv2.convexHull(max_cont, returnPoints=False)
            defects = cv2.convexityDefects(max_cont, hull)
            far_point = self.farthest_point(defects, max_cont, cnt_centroid)
            print("Centroid : " + str(cnt_centroid) + ", farthest Point : " + str(far_point))
            cv2.circle(self.img, far_point, 5, [0, 0, 255], -1)
            if len(self.traverse_point) < 20:
                self.traverse_point.append(far_point)
            else:
                self.traverse_point.pop(0)
                self.traverse_point.append(far_point)

            self.draw_circles(self.traverse_point)
