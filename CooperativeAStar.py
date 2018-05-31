"""
Construct a CooperativeAStar class to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import heapq

from FeatureExtraction import *
from basics import *


class CooperativeAStar:
    def __init__(self, image, model, eta, tau):
        self.IMAGE = image
        self.MODEL = model
        self.DIST_METRIC = eta[0]
        self.DIST_VAL = eta[1]
        self.TAU = tau

        feature_extraction = FeatureExtraction(pattern='grey-box')
        self.PARTITIONS = feature_extraction.get_partitions(self.IMAGE, self.MODEL, num_partition=10)

        self.ROOT = 0
        self.NODES = {str(self.ROOT): None}
        self.ATOMIC_MANIPULATIONS = {}

    def cal_distance(self, image1, image2):
        if self.DIST_METRIC == 'L0':
            return l0Distance(image1, image2)
        elif self.DIST_METRIC == 'L1':
            return l1Distance(image1, image2)
        elif self.DIST_METRIC == 'L2':
            return l2Distance(image1, image2)
        else:
            print("Unrecognised distance metric. "
                  "Try 'L0', 'L1', 'L2'.")

    def player1(self, node):
        player1_nodes = {}
        for key in self.PARTITIONS.keys():
            player1_nodes.update({node.path + '-' + str(key): None})

        print("Player I.")

    def player2(self, node, key):
        pixels = self.PARTITIONS[key]

        player2_nodes = {}
        for index in len(pixels):
            player2_nodes.update({node.path + '-' + str(index): pixels[index]})

        print("Player II.")

    def atomic_manipulation(self):
        pixels = self.PARTITIONS[0]
        tau = self.TAU
        model = self.MODEL.model
        image = self.IMAGE

        (row, col, chl) = image.shape
        img_batch = np.kron(np.ones((chl * 2, 1, 1, 1)), image)

        manipulated_images = []
        for (x, y) in pixels:
            changed_img_batch = img_batch.copy()
            for z in range(chl):
                changed_img_batch[z * 2, x, y, z] += tau
                changed_img_batch[z * 2 + 1, x, y, z] -= tau
            manipulated_images.append(changed_img_batch)  # each loop append [chl*2, row, col, chl]

        manipulated_images = np.asarray(manipulated_images)  # [len(pixels), chl*2, row, col, chl]
        manipulated_images = manipulated_images.reshape(len(pixels) * chl * 2, row, col, chl)

        probabilities = model.predict(manipulated_images)

        estimation = []
        cost = []
        heuristic = []
        for idx in range(len(manipulated_images)):
            cost.append(self.cal_distance(manipulated_images[idx], self.IMAGE))
            [p_max, p_2dn_max] = heapq.nlargest(2, probabilities[idx])
            heuristic.append((p_max - p_2dn_max) * 2 / tau)
            estimation.append(cost[idx] + heuristic[idx])


