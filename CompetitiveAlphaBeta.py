#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a CompetitiveAlphaBeta class to compute
the lower bound of Player I’s maximum adversary distance
while Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""
from numpy import inf

from FeatureExtraction import *
from basics import *


class CompetitiveAlphaBeta:
    def __init__(self, image, model, eta, tau, bounds=(0, 1)):
        self.IMAGE = image
        self.IMAGE_BOUNDS = bounds
        self.MODEL = model
        self.DIST_METRIC = eta[0]
        self.DIST_VAL = eta[1]
        self.TAU = tau
        self.LABEL, _ = self.MODEL.predict(self.IMAGE)

        feature_extraction = FeatureExtraction(pattern='grey-box')
        self.PARTITIONS = feature_extraction.get_partitions(self.IMAGE, self.MODEL, num_partition=50)

        self.ALPHA = {}
        self.BETA = {}
        self.MANI_BETA = {}
        self.CURRENT_MANI = ()

        self.ADVERSARY_FOUND = None
        self.ADVERSARY = None

        print("Distance metric %s, with bound value %s." % (self.DIST_METRIC, self.DIST_VAL))

    def play_game(self, image):
        self.player1(image)

    def player1(self, image):
        # Alpha
        for partitionID in self.PARTITIONS.keys():
            self.player2(image, partitionID)
            self.MANI_BETA = {}
            self.CURRENT_MANI = ()

    def player2(self, image, partition_idx):
        # Beta
        pixels = self.PARTITIONS[partition_idx]
        self.target_pixels(image, pixels)
        self.feature_robustness(pixels, partition_idx)

    def feature_robustness(self, pixels, partition_idx):
        min_dist = min(self.MANI_BETA.values())
        if min_dist is not inf:
            print("Adversary found.")
            adv_mani = min(self.MANI_BETA, key=self.MANI_BETA.get)
            adv_dist = self.MANI_BETA[adv_mani]
            self.BETA.update({partition_idx: [adv_mani, adv_dist]})
        else:
            print("Adversary not found.")
            mani_distance = copy.deepcopy(self.MANI_BETA)
            for atom, _ in mani_distance.items():
                self.MANI_BETA.pop(atom)
                self.CURRENT_MANI = atom

                new_image = copy.deepcopy(self.IMAGE)
                atomic_list = [atom[i:i + 4] for i in range(0, len(atom), 4)]
                for atomic in atomic_list:
                    valid, new_image = self.apply_atomic_manipulation(new_image, atomic)

                self.target_pixels(new_image, pixels)

            self.feature_robustness(pixels, partition_idx)

    def target_pixels(self, image, pixels):
        (row, col, chl) = image.shape

        atomic_manipulations = []
        manipulated_images = []
        for (x, y) in pixels:
            for z in range(chl):
                atomic = (x, y, z, 1 * self.TAU)
                valid, atomic_image = self.apply_atomic_manipulation(image, atomic)
                if valid is True:
                    manipulated_images.append(atomic_image)
                    atomic_manipulations.append(atomic)
                # atomic = (x, y, z, -1 * self.TAU)
                # valid, atomic_image = self.apply_atomic_manipulation(image, atomic)
                # if valid is True:
                #     manipulated_images.append(atomic_image)
                #     atomic_manipulations.append(atomic)
        manipulated_images = np.asarray(manipulated_images)

        probabilities = self.MODEL.model.predict(manipulated_images)
        labels = probabilities.argmax(axis=1)

        for idx in range(len(manipulated_images)):
            if labels[idx] != self.LABEL:
                dist = self.cal_distance(manipulated_images[idx], self.IMAGE)
                self.MANI_BETA.update({self.CURRENT_MANI + atomic_manipulations[idx]: dist})
            else:
                self.MANI_BETA.update({self.CURRENT_MANI + atomic_manipulations[idx]: inf})

    def apply_atomic_manipulation(self, image, atomic):
        atomic_image = image.copy()
        chl = atomic[0:3]
        manipulate = atomic[3]

        if (atomic_image[chl] >= max(self.IMAGE_BOUNDS) and manipulate >= 0) or (
                atomic_image[chl] <= min(self.IMAGE_BOUNDS) and manipulate <= 0):
            valid = False
            return valid, atomic_image
        else:
            if atomic_image[chl] + manipulate > max(self.IMAGE_BOUNDS):
                atomic_image[chl] = max(self.IMAGE_BOUNDS)
            elif atomic_image[chl] + manipulate < min(self.IMAGE_BOUNDS):
                atomic_image[chl] = min(self.IMAGE_BOUNDS)
            else:
                atomic_image[chl] += manipulate
            valid = True
            return valid, atomic_image

    def cal_distance(self, image1, image2):
        if self.DIST_METRIC == 'L0':
            return l0Distance(image1, image2)
        elif self.DIST_METRIC == 'L1':
            return l1Distance(image1, image2)
        elif self.DIST_METRIC == 'L2':
            return l2Distance(image1, image2)
        else:
            print("Unrecognised distance metric. "
                  "Try 'L0', 'L1', or 'L2'.")


