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


class game_moves:

    def __init__(self, data_set, model, image, tau):
        self.data_set = data_set
        self.model = model
        self.image = image 
        self.tau = tau


    def initialisation(self):
    
        imageEnlargeProportion = 1
        image1 = copy.deepcopy(self.image)
        if np.max(image1) <= 1: 
            image1 = (image1*255).astype(np.uint8)
        else: 
            image1 = image1.astype(np.uint8)
    
        if max(image1.shape) < 100 and K.backend() == 'tensorflow': 
            image1 = cv2.resize(image1, (0,0), fx=imageEnlargeProportion, fy=imageEnlargeProportion)
            kp = self.SIFT_Filtered_twoPlayer(image1)
        else: 
            #kp, des = SIFT_Filtered(image1)
            kp = self.SIFT_Filtered_twoPlayer(image1)

                  
        print("%s keypoints are found. "%(len(kp)))
    
        actions = {}
        actions[0] = kp
        s = 1
        kp2 = []
        
        if len(image1.shape) == 2: 
            image0 = np.zeros(image1.shape)
        else: 
            image0 = np.zeros(image1.shape[:2])

        numOfmanipulations = 0 
        points_all = self.getPoints_twoPlayer(image1, kp)
        print("The pixels are partitioned with respect to keypoints. ")
        for k, points in points_all.items(): 
            allRegions = []
            for i in range(len(points)):
        #     print kp[i].pt
                points[i] = (points[i][0]/imageEnlargeProportion, points[i][1]/imageEnlargeProportion)
            points = list(set(points))
            num = len(points)
            i = 0
            while i < num :
                nexttau = {}
                nextNumtau = {}    
                ls = [] 
                x = int(points[i][0])
                y = int(points[i][1])
                if image0[x][y] == 0 and len(image1.shape) == 2:  
                    ls.append((x,y))
                elif image0[x][y] == 0: 
                    ls.append((x,y,0))
                    ls.append((x,y,1))
                    ls.append((x,y,2))
                image0[x][y] = 1
            
                if len(ls) > 0: 
                    for j in ls: 
                        nexttau[j] = self.tau
                        nextNumtau[j] = 1 / self.tau            
                    oneRegion = (nexttau,nextNumtau,1)
                    allRegions.append(oneRegion)
                i += 1
            actions[s] = allRegions
            kp2.append(kp[s-1])
            s += 1
            print("%s manipulations have been initialised for keypoint (%s,%s), whose response is %s."%(len(allRegions), int(kp[k-1].pt[0]/imageEnlargeProportion), int(kp[k-1].pt[1]/imageEnlargeProportion),kp[k-1].response))
            numOfmanipulations += len(allRegions)
        actions[0] = kp2
        print("the number of all manipulations initialised: %s\n"%(numOfmanipulations))
        self.moves = actions
        
    def applyManipulation(self,image,span,numSpan):

        # toggle manipulation
        image1 = copy.deepcopy(image)
        maxVal = np.max(image1)
        minVal = np.min(image1)
        for elt in span.keys(): 
            if len(elt) == 2: 
                (fst,snd) = elt 
                if maxVal - image[fst][snd] < image[fst][snd] : image1[fst][snd] -= numSpan[elt] * span[elt]
                else: image1[fst][snd] += numSpan[elt] * span[elt]
                if image1[fst][snd] < minVal: image1[fst][snd] = minVal
                elif image1[fst][snd] > maxVal: image1[fst][snd] = maxVal
            elif len(elt) == 3: 
                (fst,snd,thd) = elt 
                if maxVal - image[fst][snd][thd] < image[fst][snd][thd] : image1[fst][snd][thd] -= numSpan[elt] * span[elt]
                else: image1[fst][snd][thd] += numSpan[elt] * span[elt]
                if image1[fst][snd][thd] < minVal: image1[fst][snd][thd] = minVal
                elif image1[fst][snd][thd] > maxVal: image1[fst][snd][thd] = maxVal
        return image1
    
        
    def SIFT_Filtered_twoPlayer(self,image): #threshold=0.0):
        sift =  cv2.xfeatures2d.SIFT_create() # cv2.SIFT() # cv2.SURF(400) #   
        kp, des = sift.detectAndCompute(image,None)
        return  kp
    
    def getPoints_twoPlayer(self,image, kps): 
        import operator
        import random
        maxNumOfPointPerKeyPoint = 100
        points = {}
        if self.data_set != "imageNet": 
            for x in range(max(image.shape)): 
                for y in range(max(image.shape)): 
                    ps = 0
                    maxk = -1
                    for i in range(1, len(kps)+1): 
                        k = kps[i-1]
                        dist2 = np.linalg.norm(np.array([x,y]) - np.array([k.pt[0],k.pt[1]]))
                   #print("aaa(%s,%s)"%(k.pt[0],k.pt[1]))
                        ps2 = norm.pdf(dist2, loc=0.0, scale=k.size)
                        if ps2 > ps: 
                            ps = ps2
                            maxk = i
                #maxk = max(ps.iteritems(), key=operator.itemgetter(1))[0]
                    if maxk in points.keys(): 
                        points[maxk].append((x,y))
                    else: points[maxk] = [(x,y)]
            if maxNumOfPointPerKeyPoint > 0: 
                for mk in points.keys():
                    beginingNum = len(points[mk])
                    for i in range(beginingNum - maxNumOfPointPerKeyPoint): 
                        points[mk].remove(random.choice(points[mk]))
            return points
        else: 
            kps = kps[:200]
            eachNum = max(image.shape) ** 2 / len(kps)
            maxk = 1
            points[maxk] = []
            for x in range(max(image.shape)): 
                for y in range(max(image.shape)): 
                    if len(points[maxk]) <= eachNum: 
                        points[maxk].append((x,y))
                    else: 
                        maxk += 1
                        points[maxk] = [(x,y)]   
            return points             
    