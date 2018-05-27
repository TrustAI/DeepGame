#!/usr/bin/env python

"""
author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy


class dataCollection:
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
        self.fileName = "dataCollection_%s.txt" % (filenamePostfix)
        self.fileHandler = open(self.fileName, 'w')
        self.maxfeatures = {}

    def initialiseIndex(self, index):
        self.index = index

    def addMaxFeatures(self, rt):
        self.maxfeatures[self.index] = rt

    def addRunningTime(self, rt):
        self.runningTime[self.index] = rt

    def addConfidence(self, cf):
        self.confidence[self.index] = cf

    def addManipulationPercentage(self, mp):
        self.manipulationPercentage[self.index] = mp

    def addl2Distance(self, eudist):
        self.l2Distance[self.index] = eudist

    def addl1Distance(self, l1dist):
        self.l1Distance[self.index] = l1dist

    def addl0Distance(self, l0dist):
        self.l0Distance[self.index] = l0dist

    def addmaxfeatures(self, maxfeatures):
        self.maxfeatures[self.index] = maxfeatures

    def addComment(self, str):
        self.fileHandler.write(str)

    def provideDetails(self):
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
        for i, r in self.maxfeatures.items():
            self.fileHandler.write("%s:%s\n" % (i, r))
        self.fileHandler.write("\n")

    def summarise(self):
        if len(self.manipulationPercentage) == 0:
            self.fileHandler.write("none of the images were successfully manipulated. ")
            return
        else:
            art = sum(self.runningTime.values()) / len(self.runningTime.values())
            self.fileHandler.write("average running time: %s\n" % (art))
            amp = sum(self.manipulationPercentage.values()) / len(self.manipulationPercentage.values())
            self.fileHandler.write("average manipulation percentage: %s\n" % (amp))
            eudist = sum(self.l2Distance.values()) / len(self.l2Distance.values())
            self.fileHandler.write("average euclidean distance: %s\n" % (eudist))
            l1dist = sum(self.l1Distance.values()) / len(self.l1Distance.values())
            self.fileHandler.write("average L1 distance: %s\n" % (l1dist))
            l0dist = sum(self.l0Distance.values()) / len(self.l0Distance.values())
            self.fileHandler.write("average L0 distance: %s\n" % (l0dist))

    def close(self):
        self.fileHandler.close()
