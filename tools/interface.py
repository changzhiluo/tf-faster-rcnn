# -*- coding: utf-8 -*-
#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

class Bird:
    def __init__(self): #类别名称和类别数目需要在这里定义
        class_name = ['__background__'] # 背景类，需要留着
        for line in open('./data/coco/annotations/class_names.txt', 'r'):
            class_name.append(line[0:-1]) #行尾有多一个字符：‘\n’,没有换行符时这里为line[0:-1]
        self.CLASS_NUM = len(class_name) - 1
        self.CLASSES = tuple(class_name) 
        self.NETWORKS = {'res101': 'res101_faster_rcnn_iter_100000.ckpt'}
        self.DATASETS= {'coco': 'coco_2014_train'}
        self.GPU_ID = '1'
        self.THRESH = 0.7 # thresh表示需要展示概率大于多少的结果
        self.SESS = None
        self.NET = None
        self.CFG = cfg

    
    # 初始化训练好的模型
    def load_model(self):
        self.CFG.TEST.HAS_RPN = True  # Use RPN for proposals
        # model path
        tfmodel = os.path.join('output', 'res101', self.DATASETS['coco'], 'default',
                                self.NETWORKS['res101'])
        # print('tfmodel is: ', tfmodel)
        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                        'our server and place them properly?').format(tfmodel + '.meta'))
        # set config
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU_ID
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # tfconfig.gpu_options.allow_growth=True
        tfconfig.gpu_options.allow_growth=False

        # init session
        sess = tf.Session(config=tfconfig)
        # load network
        net = resnetv1(num_layers=101)

        # net.create_architecture("TEST", self.CLASS_NUM + 1, tag='default', anchor_scales=[4, 8, 16, 32])
        net.create_architecture("TEST", self.CLASS_NUM + 1, tag='default', anchor_scales=self.CFG.ANCHOR_SCALES)
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        self.SESS = sess
        self.NET = net


    # 后处理和展示
    def vis_detections(self, im, result_list):

        # 显示原图
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')

        # 显示某一个类别的所有bbox
        for i in range(len(result_list)):
            bbox = result_list[i]['bbox']
            score = result_list[i]['score']
            class_name = result_list[i]['class_name']

            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=3.5))
            ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        plt.draw()

    # 预测函数
    def predict(self, sess, net, im):
        CONF_THRESH = self.THRESH
        # thresh表示需要展示概率大于多少的结果
        assert(CONF_THRESH > 0 and CONF_THRESH < 1)

        # 执行检测，得到所有bbox和相应的分数
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(sess, net, im)
        timer.toc()
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

        # 逐一统计每个类别的检测结果
        result_list = [] # 每个结果里面是一个字典
        NMS_THRESH = 0.3
        for cls_ind, cls_name in enumerate(self.CLASSES[1:]): # 遍历类别
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) == 0:
                continue
            for i in inds: # 遍历每个类别里面的每个框
                result = dict()
                bbox_ori = dets[i, :4] # x1, y1, x2, y2, 需要转为x，y，w，h
                bbox = [bbox_ori[0], bbox_ori[1], bbox_ori[2] - bbox_ori[0], bbox_ori[3] - bbox_ori[1]]
                score = dets[i, -1]

                result['bbox'] = bbox
                result['score'] = score
                result['class_name'] = cls_name

                result_list.append(result)
        return result_list
                      


if __name__ == '__main__':

    bird = Bird()
    bird.load_model() # 为了获取sess和net
    # 测试图片, 只支持3通道的图片，特殊图片如1通道或者4通道的可以用opencv转换为3通道的。
    img_path = ['/home/luochangzhi/code/tf-faster-rcnn/data/demo/1.jpg', '/home/luochangzhi/code/tf-faster-rcnn/data/demo/2.jpg']
    for i in range(len(img_path)):
        im = cv2.imread(img_path[i])
        result_list = bird.predict(bird.SESS, bird.NET, im)
        bird.vis_detections(im, result_list)

    plt.show()
