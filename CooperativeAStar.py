"""
Construct a CooperativeAStar class to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

from FeatureExtraction import *


class CooperativeAStar:
    def __init__(self, image, model, eta, tau):
        self.IMAGE = image
        self.MODEL = model
        self.DIST_METRIC = eta[0]
        self.DIST_VAL = eta[1]
        self.TAU = tau

    def player1(self):
        feature_extraction = FeatureExtraction(pattern='grey-box')
        key_points = feature_extraction.get_key_points(self.IMAGE, num_partition=10)
        partitions = feature_extraction.get_partitions(self.IMAGE, self.MODEL, num_partition=10)



        print("Player I.")
