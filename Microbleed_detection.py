#!/usr/bin/env python
# -*- coding: utf -8 -*-
# Shixiao Li & TianFu Li

import colorsys
import os
from timeit import default_timer as timer
import xml.etree.ElementTree as ET
import time
import cv2
import re
import pydicom
import SimpleITK as sitk
import shutil

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

dicom_input = './Dicom_project/'
jpg_output = './jpg_output/'
xml_path = "./VOCdevkit/VOC2007/Stander/"
# t2p_path = r'E:\IMAGE output\t2p_start_at_005_11.27_folder'

# 创建创建一个存储检测结果的dir
result_path = './result'
if not os.path.exists(result_path):
    os.makedirs(result_path)

# result如果之前存放的有文件，全部清除
for i in os.listdir(result_path):
    path_file = os.path.join(result_path, i)
    if os.path.isfile(path_file):
        os.remove(path_file)

# CMBs-output如果之前存放的有文件，全部清除
if os.path.exists(jpg_output):
    shutil.rmtree(jpg_output)

# 创建一个记录检测结果的文件
txt_path = result_path + '/result.txt'
file = open(txt_path, 'w')


def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype('uint8')
    newimg = cv2.cvtColor(newimg, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def t2p_100(img, count):
    if count > 4:
        _, image = cv2.threshold(img, 100, 255, type=cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # length = []
        for n in range(len(contours)):
            # length.append(len(contours[n]))
            points = []
            for i in range(len(contours[n])):
                point = contours[n][i][0].tolist()
                points.append(point)
            color = (255, 255, 255)
            res = cv2.fillPoly(img, [np.array(points)], color)
        img_hist = cv2.calcHist(images=[res], channels=[0], histSize=[256], mask=None, ranges=[0, 255])
        star = 0
        for k in range(50, 100, 1):
            if img_hist[k] < 400:
                star += 1
        if star < 10:
            # cv2.imshow("res", res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return res

        else:
            return image
    else:
        return img


def dicom2jpg(input_path, output_path):
    namelist = os.listdir(input_path)
    for name in namelist:
        patient_name = '%s' % name
        path = input_path + "%s" % patient_name
        outputpath_dir = []
        formats = ["Ax T2 FLAIR", "Ax T2 P", "Ax SWAN"]
        simple_name = name[:name.index('-')]
        for format in formats:
            outputpath_dir.append(output_path + '%s/%s/' % (simple_name, format))
        print(outputpath_dir)
        for outputpath in outputpath_dir:
            if not os.path.exists(outputpath):
                os.makedirs(outputpath)
        filename = os.listdir(path)
        # 用lstFilesDCM作为存放DICOM files的列表
        lstFilesDCM = []
        count = 0
        count_Flair = 0
        count_T2 = 0
        count_SWAN = 0
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
                    # print(filename)
                    lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中
                    RefDs = pydicom.read_file(os.path.join(dirName, filename))
                    # print(RefDs.SeriesDescription)
                    # print(RefDs)
                    # print(RefDs.PatientName)
                    for format in formats:
                        format_index = formats.index(format)  # 得到该format在formats中的位置
                        if format in RefDs.SeriesDescription:  # 只转换指定的序列的DICOM文件
                            if format_index == 0:
                                count_Flair = count_Flair + 1
                                count = count_Flair - 1
                            if format_index == 1:
                                count_T2 = count_T2 + 1
                                count = count_T2 - 1
                            if format_index == 2:
                                count_SWAN = count_SWAN + 1
                                count = count_SWAN - 1
                            Form = RefDs.SeriesDescription
                            patientName = str(RefDs.PatientName)
                            location = str(RefDs.SliceLocation)
                            number = str(count)
                            number_3bit = number.zfill(3)
                            document = os.path.join(path, filename)
                            countfullname = patientName + number_3bit + '_' + location + '.jpg'
                            # print(countfullname)
                            outputpath = outputpath_dir[format_index]
                            output_jpg_path = os.path.join(outputpath, countfullname)
                            ds_array = sitk.ReadImage(document)
                            img_array = sitk.GetArrayFromImage(ds_array)
                            shape = img_array.shape  # name.shape
                            img_array = np.reshape(img_array, (shape[1], shape[2]))
                            high = np.max(img_array)
                            low = np.min(img_array)
                            # convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)
                            lungwin = np.array([low * 1., high * 1.])
                            newimg = (img_array - lungwin[0]) / (lungwin[1] - lungwin[0])
                            newimg = (newimg * 255).astype('uint8')
                            if format_index == 1:
                                cv2.imwrite(output_jpg_path, t2p_100(newimg, count),
                                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                                # cv2.imwrite(output_jpg_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                            else:
                                newimg = cv2.cvtColor(newimg, cv2.COLOR_GRAY2BGR)
                                cv2.imwrite(output_jpg_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/trained_weights_final.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors // 2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
            self.yolo_model.load_weights(self.model_path)  # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                   num_anchors / len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2,))
        if self.gpu_num >= 2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                                           len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        location = []
        scores = []
        start = timer()  # 开始计时

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)  # 打印图片的尺寸
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 提示用于找到几个bbox

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        # 保存框检测出的框的个数
        # file_txt.write('find  ' + str(len(out_boxes)) + ' target(s) \n')

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            # # 写入检测位置
            # file_txt.write(
            #     predicted_class + '  score: ' + str(score) + ' \nlocation: top: ' + str(top) + '、 bottom: ' + str(
            #         bottom) + '、 left: ' + str(left) + '、 right: ' + str(right) + '\n')

            # print(label, (left, top), (right, bottom))
            loc = [top, left, bottom, right]
            scores.append(score)
            location.append(loc)
            # print(location)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        # print('time consume:%.3f s ' % (end - start))
        return image, location, scores

    def detect_target_location(self, image):
        location = []
        scores = []
        start = timer()  # 开始计时

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)  # 打印图片的尺寸
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))  # 提示用于找到几个bbox

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(2e-2 * image.size[1] + 0.2).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 500

        # 保存框检测出的框的个数
        # file_txt.write('find  ' + str(len(out_boxes)) + ' target(s) \n')

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            loc = [top, left, bottom, right]
            scores.append(score)
            location.append(loc)
        return image, location, scores

    def close_session(self):
        self.sess.close()


# 检测IoU的分数
def bboxes_intersection(bboxes_ref, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score


def detect_circle(image, location):
    """
    检测圆形程序
    :param image: 单通道灰度图
    :param location: 这里输入的是[y1,x1,y2,x2]
    :return: 返回是否真实存在圆形
    """
    global xc, yc
    # print(location)
    xr = int((location[1] + location[3]) / 2)
    yr = int((location[0] + location[2]) / 2)
    # print('矩形的中点坐标是：%s,%s' % (xr, yr))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(image, (location[1], location[0]), (location[3], location[2]), (0, 255, 0), 1)
    cropImg = image[location[1]:location[3], location[0]:location[2]]
    circles = cv2.HoughCircles(cropImg, cv2.HOUGH_GRADIENT, 1, 18, param1=5, param2=5, minRadius=1, maxRadius=20)
    if circles is not None:
        for circle in circles[0]:
            x = circle[0]
            y = circle[1]
            xc = x + location[1]
            yc = y + location[0]
            # print('圆的中点坐标是：%s,%s' % (xc, yc))
            # print('矩形的中点坐标是：%s,%s' % (xr, yr))
    if abs(xr - xc) <= 3 and abs(yr - yc) <= 3:
        return True
    else:
        return False


def operator_method(location, scores, t2p_path, points_threshold):
    t2p_dict = {}
    pattern = re.compile(r'[A-Za-z]*', re.I)
    person_name = pattern.search(name).group()
    p_location = []
    p_score = []
    # t2p_path = t2p_path + '/' + person_name + '/'
    if not os.path.exists(t2p_path):
        return location, scores
    t2p_list = os.listdir(t2p_path)
    # 原图的切片位置
    if 'shimkonis' in name:
        # 未去脑壳，第一个为2，去脑壳第一个为1
        loc = int(name.split('_')[1].split('.')[0])
    else:
        loc = int(name.split('_')[1].split('.')[0])
    # 把t2p中的切片位置和图片名建立一个字典对应起来
    for pic2 in t2p_list:
        if 'shimkonis' in pic2:
            # print(pic2)
            loc2 = int(pic2.split('_')[2].split('.')[0])
        else:
            loc2 = int(pic2.split('_')[1].split('.')[0])
        t2p_dict[loc2] = pic2
    # 找到离原图的最近的t2p
    loc_near = 999
    t2p_loc = 999
    loc2s = t2p_dict.keys()
    for l in loc2s:
        if abs(l - loc) < t2p_loc:
            t2p_loc = l - loc
            loc_near = l
    # 得到离loc最近的图片
    t2p_pic = t2p_dict[loc_near]

    for z in range(len(location)):
        star = 0
        # [t, l, b, r]
        # 边长
        length = 6
        length_half = length / 2
        middle_x = int(round((location[z][1] + location[z][3]) / 2))
        middle_y = int(round((location[z][0] + location[z][2]) / 2))
        # if use 729 , use white_name to replace name

        if not os.path.exists(t2p_path):
            continue
        t2p_img = cv2.imread(t2p_path + '/' + t2p_pic)
        if middle_x > 511 or middle_y > 511 or middle_x + length_half > 511 or middle_y + length_half > 511 or middle_x - length_half < 0 or middle_y - length_half < 0:
            continue
        # cv2对图片索引的顺序是行列顺序
        ### 矩形法#####
        point_sum = []
        color_dict = {}
        # 左上角的点向右前进5格
        for s1 in range(length):
            point1 = (middle_x - length_half + s1, middle_y - length_half)
            point_sum.append(point1)
        # 右上角的点向下前进5格
        for s2 in range(1, length):
            point2 = (middle_x + length_half, middle_y - length_half + s2)
            point_sum.append(point2)
        # 右下角的点向左前进5格
        for s3 in range(1, length):
            point3 = (middle_x + length_half - s3, middle_y + length_half)
            point_sum.append(point3)
        # 左下角的点向上前进4格
        for s4 in range(1, length - 1):
            point4 = (middle_x - length_half, middle_y + length_half - s4)
            point_sum.append(point4)
        # # 主对角线,左下到右上
        for s5 in range(1, length - 1):
            point5 = (middle_x - length_half + s5, middle_y + length_half + s5)
            point_sum.append(point5)
        # 副对角线， 中点到左上
        for s6 in range(1, length - 1):
            point6 = (middle_x - length_half + s6, middle_y - length_half + s6)
            point_sum.append(point6)

        # print('total is ', len(point_sum))

        for c in range(len(point_sum)):
            point = point_sum[c]
            color_dict[point] = t2p_img[int(point[1])][int(point[0])][0]
        for value in color_dict.values():
            if value < 200:
                star += 1
        # 共有6n-8个点
        # if star > (3*length - 5):
        if star > points_threshold:
            p_location.append(location[z])
            p_score.append(scores[z])

    location = p_location
    scores = p_score
    return location, scores


def mask_method(location, scores, t2p_path, points_threshold):
    t2p_dict = {}
    pattern = re.compile(r'[A-Za-z]*', re.I)
    person_name = pattern.search(name).group()
    p_location = []
    p_score = []
    # t2p_path = t2p_path + '/' + person_name + '/'
    if not os.path.exists(t2p_path):
        return location, scores
    t2p_list = os.listdir(t2p_path)
    # 原图的切片位置
    if 'shimkonis' in name:
        # 未去脑壳，第一个为2，去脑壳第一个为1
        loc = int(name.split('_')[1].split('.')[0])
    else:
        loc = int(name.split('_')[1].split('.')[0])
    # 把t2p中的切片位置和图片名建立一个字典对应起来
    for pic2 in t2p_list:
        if 'shimkonis' in pic2:
            # print(pic2)
            loc2 = int(pic2.split('_')[2].split('.')[0])
        else:
            loc2 = int(pic2.split('_')[1].split('.')[0])
        t2p_dict[loc2] = pic2
    # 找到离原图的最近的t2p
    loc_near = 999
    t2p_loc = 999
    loc2s = t2p_dict.keys()
    for l in loc2s:
        if abs(l - loc) < t2p_loc:
            t2p_loc = l - loc
            loc_near = l
    # 得到离loc最近的图片
    t2p_pic = t2p_dict[loc_near]

    for z in range(len(location)):
        star = 0
        # [t, l, b, r]
        # 边长
        length = 6
        length_half = length / 2
        middle_x = int(round((location[z][1] + location[z][3]) / 2))
        middle_y = int(round((location[z][0] + location[z][2]) / 2))
        # if use 729 , use white_name to replace name

        if not os.path.exists(t2p_path):
            continue
        t2p_img = cv2.imread(t2p_path + '/' + t2p_pic)
        if middle_x > 511 or middle_y > 511 or middle_x + length_half > 511 or middle_y + length_half > 511 or middle_x - length_half < 0 or middle_y - length_half < 0:
            continue
        # cv2对图片索引的顺序是行列顺序
        # 原图是img，以cv2读入的灰度图
        # t2p图是t2p_img，以cv2读入的彩图
        # location = [t, l, b, r]
        # p1 = (xmin, ymin), p2 = (xmax, ymax)
        p1 = (location[z][1], location[z][0])
        p2 = (location[z][3], location[z][2])
        original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 截取bbox区域
        crop_image = original_img[p1[1]:p2[1], p1[0]:p2[0]]
        # 创建一个白板，拼接 crop区域
        white = np.ones((512, 512, 3), dtype=np.uint8) * 255
        white[p1[1]:p2[1], p1[0]:p2[0]] = crop_image
        # 二值翻转得到黑点区域，阈值设置为100
        _, threshold_image = cv2.threshold(white, 100, 255, type=cv2.THRESH_BINARY_INV)
        threshold_image_B, threshold_image_G, threshold_image_R = cv2.split(threshold_image)
        # 与操作得到t2p的黑点区域
        t2p_threshold_image = threshold_image & t2p_img
        B, G, R = cv2.split(t2p_threshold_image)
        B_np = np.array(B)
        threshold_image_B_np = np.array(threshold_image_B)
        sum = threshold_image_B[np.where(threshold_image_B > 5)]
        white_sum = B_np[np.where(B_np == 255)]
        length_sum = len(sum)
        length_white = len(white_sum)
        if length_sum == 0:
            length_sum = 1
        # print(B_np.shape)
        # print(sum)
        # print(white_sum)
        score_white = length_white / length_sum
        if score_white < points_threshold:
            p_location.append(location[z])
            p_score.append(scores[z])

    location = p_location
    scores = p_score

    return location, scores


def Comparation(bbox1, bbox2, error=3):
    """
    判别是否是同一个CMB
    :param bbox1: 第一个框
    :param bbox2: 第二个框
    :param error: 可接受误差
    :return: bool
    """
    x0 = abs(bbox1[0] - bbox2[0])
    y0 = abs(bbox1[1] - bbox2[1])
    z0 = abs(bbox1[2] - bbox2[2])

    if x0 <= error and y0 <= error and z0 <= 2:
        return True
    else:
        return False


def hierarchy(CMB):
    # 先对z轴进行排序
    CMB_z_sorted = sorted(CMB, key=lambda z: z[2])
    # print("对z轴排序后为：%s" % CMB_z_sorted)
    # 看z轴出现哪些层
    only_z_axis = np.array(CMB_z_sorted)
    # print("只取z轴得到的数据%s" % only_z_axis[::, 2])
    z_axis = np.unique(only_z_axis[::, 2])
    # print("微出血所在的图层%s" % z_axis)
    structure_map = []
    temp = []
    # 把CMB数组按照z轴的图片层关系，切分出来
    for pic_slice in range(len(z_axis)):
        if CMB_z_sorted is not None:
            # 将numpy 矩阵转换成列表
            z_axis_list = z_axis.tolist()
            # 该层出现的微出血点数目
            CMBs_pic = only_z_axis[::, 2].tolist().count(z_axis_list[pic_slice])
            # print("当前层有%s个点" % CMBs_pic)
            for count in range(CMBs_pic):
                temp.append(CMB_z_sorted[0])
                CMB_z_sorted.remove(CMB_z_sorted[0])
            structure_map.append(temp)
            temp = []
    # print("分层结果:%s" % structure_map)
    return structure_map


def CMBs_detection(structure_map):
    CMB_number = []
    False_count = 0
    Temp_count = 0
    for floor in range(len(structure_map)):
        # print("进行到第%s层" % floor)
        if floor == 0:
            for temp_CMB_number in range(len(structure_map[floor])):
                temp_number = structure_map[0][temp_CMB_number]
                CMB_number.append([temp_number])
            # print("初步建立微出血个数%s" % CMB_number)
        else:
            for current_CMB in range(len(structure_map[floor])):
                for CMBs in range(len(CMB_number)):
                    if Comparation(CMB_number[CMBs][-1], structure_map[floor][current_CMB]) is True:
                        CMB_number[CMBs].append(structure_map[floor][current_CMB])
                    else:
                        False_count = False_count + 1
                    # print(CMB_number)
                    # print(CMB_number[CMBs][-1], structure_map[floor][current_CMB],
                    #       Comparation(CMB_number[CMBs][-1], structure_map[floor][current_CMB]))
                    if False_count == len(CMB_number):
                        CMB_number.append([structure_map[floor][current_CMB]])
                        False_count = 0
                False_count = 0
    for number in range(len(CMB_number)):
        if len(CMB_number[number]) == 1:
            Temp_count = Temp_count + 1
    final_number = (len(CMB_number) - Temp_count)
    # print("最终微出血情况%s" % CMB_number)
    # print("最终微出血%s" % np.array(CMB_number))
    return final_number

def CMBs_burden(CMB):
    if len(CMB) > 0:
        return CMBs_detection(hierarchy(CMB))
    else:
        return 0
# 图片检测

if __name__ == '__main__':
    print("数据准备中...")
    dicom2jpg(dicom_input, jpg_output)
    print("数据准备完成...")
    print("系统开始测试")
    t1 = time.time()

    true_positive = 0
    all_detection = 0
    all_ground_truths = 0
    location_medial = []

    patient_name = os.listdir(jpg_output)
    for patient in range(len(patient_name)):
        print('第%s个测试对象' % (patient + 1))
        print(patient_name[patient])
        count = 0
        path = jpg_output + patient_name[patient] + '/Ax SWAN/'
        t2p_path = jpg_output + patient_name[patient] + '/Ax T2 P/'
        image_number = len(os.listdir(path))
        pic_list = os.listdir(path)
        xml_list = os.listdir(xml_path)
        yolo = YOLO()
        for i, name in enumerate(pic_list):
            xml_name = name[:-4] + '.xml'
            file_name = os.listdir(path)[i]
            image = Image.open(path + file_name)
            img = cv2.imread(path + file_name, 0)
            draw_rect = ImageDraw.Draw(image)
            if xml_name in xml_list:
                tree = ET.parse(xml_path + '/' + xml_name)
                root = tree.getroot()
            # file.write(file_name + ' detect_result：\n')
            r_image, location, scores = yolo.detect_image(image)
            # 写入坐标到txt文档中
            # if len(location) > 0:
            #     file.write(str(location))
            ########## check the point in white pictures##########
            if len(location) > 0:
                ad_original = len(location)
                ### operator method ###
                location, scores = operator_method(location, scores, t2p_path, points_threshold=10)
                ### mask method ###
                # location, scores = mask_method(location, scores, t2p_path, points_threshold=0.31)
                for count_i in range(len(location)):
                    x0 = (location[count_i][2] + location[count_i][0]) / 2
                    y0 = (location[count_i][3] + location[count_i][1]) / 2
                    z0 = i
                    location_medial.append([x0, y0, z0])
            ########## end ##########
            """ 读取xml列表 """
            if xml_name in xml_list:
                for bbox in root:
                    if bbox.tag == 'object':
                        xmin = int(bbox[4][0].text)
                        ymin = int(bbox[4][1].text)
                        xmax = int(bbox[4][2].text)
                        ymax = int(bbox[4][3].text)
                        bboxes_ref = [ymin, xmin, ymax, xmax]
                        """画个标准点的框"""
                        # draw_rect.rectangle([xmin, ymin, xmax, ymax])
                        all_ground_truths = all_ground_truths + 1
                        if len(location) is not 0:
                            for z in range(len(location)):
                                score = bboxes_intersection(bboxes_ref, location[z])
                                if score > 0.5:
                                    true_positive = true_positive + 1
                                    # if detect_circle(img, location[z]) is True:
                                    #     true_positive = true_positive + 1
            """ 模型检测到的微出血点 """
            if len(location) is not 0:
                """ 存储图片 """
                file.write('\n')
                image_save_path = './result/' + name
                r_image.save(image_save_path)
                for z in range(len(location)):
                    all_detection = all_detection + 1

            count = count + 1
            percentage = (count / image_number) * 100
            # print('测试已经进行到：%f%%' % percentage)
        print("微出血个数是：%s" % CMBs_burden(location_medial))
        file.write(str(patient_name[patient]))
        file.write('The number of microbleed is : ' + str(CMBs_burden(location_medial)))
        location_medial = []
        print('TP is :%s' % true_positive)
        print('all detection is :%s' % all_detection)
        print('all ground truths is :%s' % all_ground_truths)
        print('--------------------------------------------')
        precision = (true_positive / all_detection)
        recall = (true_positive / all_ground_truths)
        print('precision is :%s' % precision)
        print('recall is :%s' % recall)
        time_sum = time.time() - t1
        print('time sum:%ss' % time_sum)
        # file.write('time sum: ' + str(time_sum) + 's')
        print('============================================')
file.close()
# yolo.close_session()
