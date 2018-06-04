#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy

from basics import assure_path_exists


class DataCollection:
    index = 0
    layer = 0
    fileHandler = 0

    def __init__(self, filenamePostfix):
        self.runningTime = {}
        self.manipulationPercentage = {}
        self.l2Distance = {}
        self.l1Distance = {}
        self.l0Distance = {}
        self.confidence = {}
        assure_path_exists("dataCollection/")
        self.fileName = "dataCollection/%s.txt" % filenamePostfix
        self.fileHandler = open(self.fileName, 'a')
        self.maxFeatures = {}

    def initialiseIndex(self, index):
        self.index = index

    def addMaxFeatures(self, max_feature):
        self.maxFeatures[self.index] = max_feature

    def addRunningTime(self, running_time):
        self.runningTime[self.index] = running_time

    def addConfidence(self, confidence):
        self.confidence[self.index] = confidence

    def addManipulationPercentage(self, mani_percentage):
        self.manipulationPercentage[self.index] = mani_percentage

    def addl2Distance(self, l2dist):
        self.l2Distance[self.index] = l2dist

    def addl1Distance(self, l1dist):
        self.l1Distance[self.index] = l1dist

    def addl0Distance(self, l0dist):
        self.l0Distance[self.index] = l0dist

    def addComment(self, comment):
        self.fileHandler.write(comment)

    def provideDetails(self):
        if not bool(self.maxFeatures):
            self.fileHandler.write("running time: \n")
            for i, r in self.runningTime.items():
                self.fileHandler.write("%s:%s\n" % (i, r))

            self.fileHandler.write("manipulation percentage: \n")
            for i, r in self.manipulationPercentage.items():
                self.fileHandler.write("%s:%s\n" % (i, r))

            self.fileHandler.write("L2 distance: \n")
            for i, r in self.l2Distance.items():
                self.fileHandler.write("%s:%s\n" % (i, r))

            self.fileHandler.write("L1 distance: \n")
            for i, r in self.l1Distance.items():
                self.fileHandler.write("%s:%s\n" % (i, r))

            self.fileHandler.write("L0 distance: \n")
            for i, r in self.l0Distance.items():
                self.fileHandler.write("%s:%s\n" % (i, r))

            self.fileHandler.write("confidence: \n")
            for i, r in self.confidence.items():
                self.fileHandler.write("%s:%s\n" % (i, r))
            self.fileHandler.write("\n")

            self.fileHandler.write("max features: \n")
            for i, r in self.maxFeatures.items():
                self.fileHandler.write("%s:%s\n" % (i, r))
            self.fileHandler.write("\n")
        else:
            self.fileHandler.write("none of the inputs were successfully manipulated")

    def summarise(self):
        if len(self.manipulationPercentage) is 0:
            self.fileHandler.write("none of the images were successfully manipulated.")
            return
        else:
            # art = sum(self.runningTime.values()) / len(self.runningTime.values())
            art = np.mean(list(self.runningTime.values()))
            self.fileHandler.write("average running time: %s\n" % art)
            # amp = sum(self.manipulationPercentage.values()) / len(self.manipulationPercentage.values())
            amp = np.mean(list(self.manipulationPercentage.values()))
            self.fileHandler.write("average manipulation percentage: %s\n" % amp)
            # l2dist = sum(self.l2Distance.values()) / len(self.l2Distance.values())
            l2dist = np.mean(list(self.l2Distance.values()))
            self.fileHandler.write("average euclidean distance: %s\n" % l2dist)
            # l1dist = sum(self.l1Distance.values()) / len(self.l1Distance.values())
            l1dist = np.mean(list(self.l1Distance.values()))
            self.fileHandler.write("average L1 distance: %s\n" % l1dist)
            # l0dist = sum(self.l0Distance.values()) / len(self.l0Distance.values())
            l0dist = np.mean(list(self.l0Distance.values()))
            self.fileHandler.write("average L0 distance: %s\n\n\n\n\n" % l0dist)

    def close(self):
        self.fileHandler.close()
