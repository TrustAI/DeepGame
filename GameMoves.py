#!/usr/bin/env python

"""
author: Xiaowei Huang

"""

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from keras import backend as K
from scipy.stats import truncnorm, norm

from basics import *
from FeatureExtraction import *
import collections


############################################################
#
#  initialise possible moves for a two-player game
#
################################################################


class GameMoves:

    def __init__(self, data_set, model, image, tau):
        self.data_set = data_set
        self.model = model
        self.image = image
        self.tau = tau

        feature_extraction = FeatureExtraction(pattern='grey-box')
        kps = feature_extraction.get_key_points(self.image, num_partition=10)
        partitions = feature_extraction.get_partitions(self.image, self.model, num_partition=10)

        img_enlarge_ratio = 1
        image1 = copy.deepcopy(self.image)
        # if np.max(image1) <= 1:
        #     image1 = (image1 * 255).astype(np.uint8)
        # else:
        #     image1 = image1.astype(np.uint8)
        #
        # if max(image1.shape) < 100:
        #     # for small images, sift works by enlarging the images
        #     image1 = cv2.resize(image1, (0, 0), fx=img_enlarge_ratio, fy=img_enlarge_ratio)
        #     kps = self.SIFT_Filtered_twoPlayer(image1)
        #     for i in range(len(kps)):
        #         oldpt = (kps[i].pt[0], kps[i].pt[1])
        #         kps[i].pt = (int(oldpt[0] / img_enlarge_ratio), int(oldpt[1] / img_enlarge_ratio))
        # else:
        #     kps = self.SIFT_Filtered_twoPlayer(image1)
        #
        # print("%s keypoints are found. " % (len(kps)))

        actions = dict()
        actions[0] = kps
        s = 1
        kp2 = []

        if len(image1.shape) == 2:
            image0 = np.zeros(image1.shape)
        else:
            image0 = np.zeros(image1.shape[:2])

        # to compute a partition of the pixels, for an image classification task 
        # partitions = self.getPartition(image1, kps)
        print("The pixels are partitioned with respect to keypoints.")

        # construct moves according to the obtained the partitions 
        num_of_manipulations = 0
        for k, blocks in partitions.items():
            all_atomic_manipulations = []

            for i in range(len(blocks)):
                x = blocks[i][0]
                y = blocks[i][1]

                (_, _, chl) = image1.shape

                # + tau 
                if image0[x][y] == 0:

                    atomic_manipulation = dict()
                    for j in range(chl):
                        atomic_manipulation[(x, y, j)] = self.tau
                    all_atomic_manipulations.append(atomic_manipulation)

                    atomic_manipulation = dict()
                    for j in range(chl):
                        atomic_manipulation[(x, y, j)] = -1 * self.tau
                    all_atomic_manipulations.append(atomic_manipulation)

                image0[x][y] = 1

            # actions[k] = all_atomic_manipulations
            actions[s] = all_atomic_manipulations
            kp2.append(kps[s - 1])

            s += 1
            # print("%s manipulations have been initialised for keypoint (%s,%s), whose response is %s."
            #       % (len(all_atomic_manipulations), int(kps[k - 1].pt[0] / img_enlarge_ratio),
            #          int(kps[k - 1].pt[1] / img_enlarge_ratio), kps[k - 1].response))
            num_of_manipulations += len(all_atomic_manipulations)

        # index-0 keeps the keypoints, actual actions start from 1
        actions[0] = kp2
        print("the number of all manipulations initialised: %s\n" % num_of_manipulations)
        self.moves = actions

    def applyManipulation(self, image, manipulation):
        # apply a specific manipulation to have a manipulated input
        image1 = copy.deepcopy(image)
        maxVal = np.max(image1)
        minVal = np.min(image1)
        for elt in list(manipulation.keys()):
            (fst, snd, thd) = elt
            image1[fst][snd][thd] += manipulation[elt]
            if image1[fst][snd][thd] < minVal:
                image1[fst][snd][thd] = minVal
            elif image1[fst][snd][thd] > maxVal:
                image1[fst][snd][thd] = maxVal
        return image1


"""
    def SIFT_Filtered_twoPlayer(self, image):  # threshold=0.0):
        sift = cv2.xfeatures2d.SIFT_create()  # cv2.SIFT() # cv2.SURF(400) #
        kp, des = sift.detectAndCompute(image, None)
        return kp

    def getPartition(self, image, kps):
        # get partition by keypoints
        import operator
        import random
        max_num_of_pixels_per_key_point = 1000000
        blocks = {}
        if self.data_set != "imageNet":
            for x in range(max(image.shape)):
                for y in range(max(image.shape)):
                    ps = 0
                    maxk = -1
                    for i in range(1, len(kps) + 1):
                        k = kps[i - 1]
                        dist2 = np.linalg.norm(np.array([x, y]) - np.array([k.pt[0], k.pt[1]]))
                        ps2 = norm.pdf(dist2, loc=0.0, scale=k.size)
                        if ps2 > ps:
                            ps = ps2
                            maxk = i
                    if maxk in blocks.keys():
                        blocks[maxk].append((x, y))
                    else:
                        blocks[maxk] = [(x, y)]
            if max_num_of_pixels_per_key_point > 0:
                for mk in blocks.keys():
                    begining_num = len(blocks[mk])
                    for i in range(begining_num - max_num_of_pixels_per_key_point):
                        blocks[mk].remove(random.choice(blocks[mk]))
            return blocks
        else:
            kps = kps[:200]
            eachNum = max(image.shape) ** 2 / len(kps)
            maxk = 1
            blocks[maxk] = []
            for x in range(max(image.shape)):
                for y in range(max(image.shape)):
                    if len(blocks[maxk]) <= eachNum:
                        blocks[maxk].append((x, y))
                    else:
                        maxk += 1
                        blocks[maxk] = [(x, y)]
            return blocks
"""
