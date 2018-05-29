"""
Construct a FeatureExtraction class to retrieve
'key points' and 'partitions' of an image
in a black-box or white-box pattern.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import copy
import numpy as np
import cv2
from scipy.stats import norm
import random

IMAGE_SIZE_BOUND = 100
MAX_NUM_OF_PIXELS_PER_KEY_POINT = 1000000


class FeatureExtraction:
    def __init__(self, image, pattern='black-box', model=None):
        self.IMAGE = image
        self.MODEL = model
        self.PATTERN = pattern

        if max(self.IMAGE.shape) < IMAGE_SIZE_BOUND:
            self.IMAGE_SIZE = 'small'
        else:
            self.IMAGE_SIZE = 'large'

        self.img_enlarge_ratio = 1

        if self.PATTERN == 'white-box' and self.MODEL is None:
            print("For 'white-box' feature extraction, please specify a neural network.")
            exit

    def get_partitions(self):
        if self.PATTERN == 'black-box':
            print("Extracting image features using '%s' pattern." % self.PATTERN)

            key_points = self.get_key_points()
            print("%s keypoints are found. " % (len(key_points)))

            partitions = {}
            if self.IMAGE_SIZE == 'small':
                for x in range(max(self.IMAGE.shape)):
                    for y in range(max(self.IMAGE.shape)):
                        ps = 0
                        maxk = -1
                        for i in range(1, len(key_points) + 1):
                            k = key_points[i - 1]
                            dist2 = np.linalg.norm(np.array([x, y]) - np.array([k.pt[0], k.pt[1]]))
                            ps2 = norm.pdf(dist2, loc=0.0, scale=k.size)
                            if ps2 > ps:
                                ps = ps2
                                maxk = i
                        if maxk in partitions.keys():
                            partitions[maxk].append((x, y))
                        else:
                            partitions[maxk] = [(x, y)]
                if MAX_NUM_OF_PIXELS_PER_KEY_POINT > 0:
                    for mk in partitions.keys():
                        begining_num = len(partitions[mk])
                        for i in range(begining_num - MAX_NUM_OF_PIXELS_PER_KEY_POINT):
                            partitions[mk].remove(random.choice(partitions[mk]))
                return partitions
            else:
                key_points = key_points[:200]
                each_num = max(self.IMAGE.shape) ** 2 / len(key_points)
                maxk = 1
                partitions[maxk] = []
                for x in range(max(self.IMAGE.shape)):
                    for y in range(max(self.IMAGE.shape)):
                        if len(partitions[maxk]) <= each_num:
                            partitions[maxk].append((x, y))
                        else:
                            maxk += 1
                            partitions[maxk] = [(x, y)]
                return partitions

        elif self.PATTERN == 'white-box':
            print("Extracting image features using '%s' pattern." % self.PATTERN)




        else:
            print("Unrecognised feature extraction pattern. "
                  "Try 'black-box' or 'white-box'.")

    def get_key_points(self):
        image = copy.deepcopy(self.IMAGE)

        sift = cv2.xfeatures2d.SIFT_create()  # cv2.SIFT() # cv2.SURF(400)

        if np.max(image) <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

        if self.IMAGE_SIZE == 'small':
            # for a small image, SIFT works by enlarging the image
            image = cv2.resize(image, (0, 0), fx=self.img_enlarge_ratio, fy=self.img_enlarge_ratio)
            key_points, _ = sift.detectAndCompute(image, None)
            for i in range(len(key_points)):
                old_pt = (key_points[i].pt[0], key_points[i].pt[1])
                key_points[i].pt = (int(old_pt[0] / self.img_enlarge_ratio), int(old_pt[1] / self.img_enlarge_ratio))
        else:
            key_points, _ = sift.detectAndCompute(image, None)

        return key_points
