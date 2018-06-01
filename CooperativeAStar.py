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
        self.CLASS, _ = self.MODEL.predict(self.IMAGE)

        feature_extraction = FeatureExtraction(pattern='grey-box')
        self.PARTITIONS = feature_extraction.get_partitions(self.IMAGE, self.MODEL, num_partition=10)

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

    def player1(self, image):
        estimations = []
        manipulations = []
        indices = []
        for partitionID in self.PARTITIONS.keys():
            estimation, manipulation, idx = self.player2(image, partitionID)
            estimations.append(estimation)
            manipulations.append(manipulation)
            indices.append(idx)

        min_partition_idx = np.argmin(estimations)
        min_manipulation = manipulations[min_partition_idx]
        min_idx = indices[min_partition_idx]

        new_image = image.copy()
        new_image[min_manipulation[0:3]] += min_manipulation[3]

        new_class, confidence = self.MODEL.predict(new_image)
        if self.cal_distance(self.IMAGE, new_image) > self.DIST_VAL:
            print("Adversarial distance exceeds distance bound.")
        elif new_class != self.CLASS:
            print("Adversarial image is found.")
        else:
            self.player1(new_image)

        print("Player I.")

    def player2(self, image, partition_idx):
        (row, col, chl) = image.shape

        pixels = self.PARTITIONS[partition_idx]
        atomic_manipulations, estimation = self.atomic_manipulation(pixels)

        min_estimation = np.min(estimation)
        min_idx = np.argmin(estimation)
        min_manipulation = atomic_manipulations[min_idx]
        # quotient, _ = divmod(min_idx, chl * 2)
        # min_pixel = pixels[quotient]

        print("Player II.")

        return min_estimation, min_manipulation, min_idx

    def atomic_manipulation(self, pixels):
        # pixels = self.PARTITIONS[0]
        tau = self.TAU
        model = self.MODEL
        image = self.IMAGE

        atomic_manipulations = {}
        (row, col, chl) = image.shape
        img_batch = np.kron(np.ones((chl * 2, 1, 1, 1)), image)

        manipulated_images = []
        idx = 0
        for (x, y) in pixels:
            changed_img_batch = img_batch.copy()
            for z in range(chl):
                changed_img_batch[z * 2, x, y, z] += tau
                atomic_manipulations.update({idx: (x, y, z, tau)})
                idx += 1
                changed_img_batch[z * 2 + 1, x, y, z] -= tau
                atomic_manipulations.update({idx: (x, y, z, -tau)})
                idx += 1
            manipulated_images.append(changed_img_batch)  # each loop append [chl*2, row, col, chl]

        manipulated_images = np.asarray(manipulated_images)  # [len(pixels), chl*2, row, col, chl]
        manipulated_images = manipulated_images.reshape(len(pixels) * chl * 2, row, col, chl)

        # probabilities = model.predict(manipulated_images)
        softmax_logits = model.softmax_logits(manipulated_images)

        estimation = []
        cost = []
        heuristic = []
        for idx in range(len(manipulated_images)):
            cost.append(self.cal_distance(manipulated_images[idx], self.IMAGE))
            [p_max, p_2dn_max] = heapq.nlargest(2, softmax_logits[idx])
            heuristic.append((p_max - p_2dn_max) * 2 / tau)
            estimation.append(cost[idx] + heuristic[idx])

        print("Atomic manipulations done.")
        return atomic_manipulations, estimation
