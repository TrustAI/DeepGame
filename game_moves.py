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
import collections


############################################################
#
#  initialise possible moves for two-player game
#
################################################################


class GameMoves:

    def __init__(self, data_set, model, image, tau):
        self.data_set = data_set
        self.model = model
        self.image = image
        self.tau = tau

        imageEnlargeProportion = 1
        image1 = copy.deepcopy(self.image)
        if np.max(image1) <= 1:
            image1 = (image1 * 255).astype(np.uint8)
        else:
            image1 = image1.astype(np.uint8)

        if max(image1.shape) < 100:
            # for small images, sift works by enlarging the images
            image1 = cv2.resize(image1, (0, 0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
            kps = self.SIFT_Filtered_twoPlayer(image1)
            for i in range(len(kps)):
                oldpt = (kps[i].pt[0], kps[i].pt[1])
                kps[i].pt = (int(oldpt[0] / imageEnlargeProportion), int(oldpt[1] / imageEnlargeProportion))
        else:
            kps = self.SIFT_Filtered_twoPlayer(image1)

        print("%s keypoints are found. " % (len(kps)))

        actions = {}
        actions[0] = kps
        s = 1
        kp2 = []

        if len(image1.shape) == 2:
            image0 = np.zeros(image1.shape)
        else:
            image0 = np.zeros(image1.shape[:2])

        # to compute a partition of the pixels, for an image classification task 
        partitions = self.getPartition(image1, kps)
        print("The pixels are partitioned with respect to keypoints. ")

        # construct moves according to the obtained the partitions 
        numOfmanipulations = 0
        for k, blocks in partitions.items():
            allAtomicManipulations = []

            for i in range(len(blocks)):
                x = blocks[i][0]
                y = blocks[i][1]

                # + tau 
                if image0[x][y] == 0 and len(image1.shape) == 2:
                    atomicManipulation = {}
                    atomicManipulation[(x, y)] = self.tau
                    allAtomicManipulations.append(atomicManipulation)
                elif image0[x][y] == 0:
                    atomicManipulation = {}
                    atomicManipulation[(x, y, 0)] = self.tau
                    allAtomicManipulations.append(atomicManipulation)
                    atomicManipulation = {}
                    atomicManipulation[(x, y, 1)] = self.tau
                    allAtomicManipulations.append(atomicManipulation)
                    atomicManipulation = {}
                    atomicManipulation[(x, y, 2)] = self.tau
                    allAtomicManipulations.append(atomicManipulation)

                # - tau   
                if image0[x][y] == 0 and len(image1.shape) == 2:
                    atomicManipulation = {}
                    atomicManipulation[(x, y)] = -1 * self.tau
                    allAtomicManipulations.append(atomicManipulation)
                elif image0[x][y] == 0:
                    atomicManipulation = {}
                    atomicManipulation[(x, y, 0)] = -1 * self.tau
                    # allAtomicManipulations.append(atomicManipulation)
                    # atomicManipulation = {}
                    atomicManipulation[(x, y, 1)] = -1 * self.tau
                    # allAtomicManipulations.append(atomicManipulation)
                    # atomicManipulation = {}
                    atomicManipulation[(x, y, 2)] = -1 * self.tau
                    allAtomicManipulations.append(atomicManipulation)

                image0[x][y] = 1

            actions[s] = allAtomicManipulations
            kp2.append(kps[s - 1])

            s += 1
            print("%s manipulations have been initialised for keypoint (%s,%s), whose response is %s." % (
            len(allAtomicManipulations), int(kps[k - 1].pt[0] / imageEnlargeProportion),
            int(kps[k - 1].pt[1] / imageEnlargeProportion), kps[k - 1].response))
            numOfmanipulations += len(allAtomicManipulations)

        # index-0 keeps the keypoints, actual actions start from 1
        actions[0] = kp2
        print("the number of all manipulations initialised: %s\n" % (numOfmanipulations))
        self.moves = actions

    def applyManipulation(self, image, manipulation):

        # apply a specific manipulation to have a manipulated input
        image1 = copy.deepcopy(image)
        maxVal = np.max(image1)
        minVal = np.min(image1)
        for elt in manipulation.keys():
            if len(elt) == 2:
                (fst, snd) = elt
                image1[fst][snd] += manipulation[elt]
                if image1[fst][snd] < minVal:
                    image1[fst][snd] = minVal
                elif image1[fst][snd] > maxVal:
                    image1[fst][snd] = maxVal
            elif len(elt) == 3:
                (fst, snd, thd) = elt
                image1[fst][snd][thd] += manipulation[elt]
                if image1[fst][snd][thd] < minVal:
                    image1[fst][snd][thd] = minVal
                elif image1[fst][snd][thd] > maxVal:
                    image1[fst][snd][thd] = maxVal
        return image1

    def SIFT_Filtered_twoPlayer(self, image):  # threshold=0.0):
        sift = cv2.xfeatures2d.SIFT_create()  # cv2.SIFT() # cv2.SURF(400) #
        kp, des = sift.detectAndCompute(image, None)
        return kp

    def getPartition(self, image, kps):
        # get partition by keypoints
        import operator
        import random
        maxNumOfPixelsPerKeyPoint = 1000000
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
            if maxNumOfPixelsPerKeyPoint > 0:
                for mk in blocks.keys():
                    beginingNum = len(blocks[mk])
                    for i in range(beginingNum - maxNumOfPixelsPerKeyPoint):
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
