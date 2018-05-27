#!/usr/bin/env python

"""
A data structure for organising search

author: Xiaowei Huang
"""

import numpy as np
import time
import os
import copy
import sys
import operator
import random
import math


from basics import *
from game_moves import *


MCTS_multi_samples = 1
effectiveConfidenceWhenChanging = 0.0
explorationRate = math.sqrt(2)        


class mcts:

    def __init__(self, data_set, model, image_index, image, gameType, tau, eta):
        self.data_set = data_set
        self.image_index = image_index
        self.image = image
        self.model = model
        self.gameType = gameType
        self.tau = tau
        self.eta = eta
        
        (self.originalClass,self.originalConfident) = self.model.predict(self.image)
        
        self.moves = game_moves(self.data_set, self.model, self.image, self.tau)

        self.cost = {}
        self.numberOfVisited = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        
        self.indexToNow = 0
        # current root node
        self.rootIndex = 0
                
        self.manipulation = {}
        # initialise root node
        self.manipulation[-1] = {}
        self.initialiseLeafNode(0,-1,{})
        
        # record all the keypoints: index -> kp
        self.keypoints = {}
        # mapping nodes to keyponts
        self.keypoint = {}
        self.keypoint[0] = 0
        
        # local actions
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}

        # best case
        self.bestCase = (2^20,{})
        self.numConverge = 0 
        
        # number of adversarial exmaples
        self.numAdv = 0
                
        # temporary variables for sampling 
        self.atomicManipulationPath = []
        self.depth = 0
        self.availableActionIDs = []
        self.usedActionIDs = [] 
        

    def initialiseMoves(self): 
        # initialise actions according to the type of manipulations
        actions = self.moves.moves
        self.keypoints[0] = 0
        i = 1 
        for k in actions[0]: 
            self.keypoints[i] = k 
            i += 1
            
        for i in range(len(actions)): 
            ast = {}
            for j in range(len(actions[i])):
                ast[j] = actions[i][j]
            self.actions[i] = ast
        nprint("%s actions have been initialised. "%(len(self.actions)))
        
    def initialiseLeafNode(self,index,parentIndex,newAtomicManipulation):
        nprint("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.manipulation[index] = mergeTwoDicts(self.manipulation[parentIndex],newAtomicManipulation)
        self.cost[index] = 0
        self.parent[index] = parentIndex 
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0    
        #activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])


    def destructor(self): 
        self.image = 0
        self.image = 0
        self.model = 0
        self.model = 0
        self.manipulatedDimensions = {}
        self.manipulation = {}
        self.cost = {}
        self.parent = {}
        self.children = {}
        self.fullyExpanded = {}
        self.numberOfVisited = {}
        
        self.actions = {}
        self.usedActionsID = {}
        self.indexToActionID = {}
        
    # move one step forward
    # it means that we need to remove children other than the new root
    def makeOneMove(self,newRootIndex): 
        if self.keypoint[newRootIndex] != 0: 
            player = "the first player"
        else: 
            player = "the second player"
        print("%s making a move into the new root %s, whose value is %s and visited number is %s"%(player, newRootIndex,self.cost[newRootIndex],self.numberOfVisited[newRootIndex]))
        self.removeChildren(self.rootIndex,[newRootIndex])
        self.rootIndex = newRootIndex
    
    def removeChildren(self,index,indicesToAvoid): 
        if self.fullyExpanded[index] == True: 
            for childIndex in self.children[index]: 
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex,[])
        self.manipulation.pop(index,None)
        self.cost.pop(index,None) 
        self.parent.pop(index,None) 
        self.keypoint.pop(index,None) 
        self.children.pop(index,None) 
        self.fullyExpanded.pop(index,None)
        self.numberOfVisited.pop(index,None)
            
    def bestChild(self,index):
        allValues = {}
        for childIndex in self.children[index]: 
            allValues[childIndex] = self.cost[childIndex] / float(self.numberOfVisited[childIndex])
        nprint("finding best children from %s"%(allValues))
        if self.gameType == "competitive" and self.keypoint[index] == 0: 
            return max(allValues.items(), key=operator.itemgetter(1))[0]
        else: 
            return min(allValues.items(), key=operator.itemgetter(1))[0]
        
    def treeTraversal(self,index):
        if self.fullyExpanded[index] == True: 
            nprint("tree traversal on node %s with childrens %s"%(index, self.children[index]))
            allValues = {}
            for childIndex in self.children[index]: 
                # UCB values
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + explorationRate * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))

            if self.gameType == "competitive" and self.keypoint[index] == 0 :
                nextIndex = np.random.choice(list(allValues.keys()), 1, p = [ x/sum(allValues.values()) for x in allValues.values()])[0] 
            else: 
                allValues2 = {}
                for k,v in allValues.items(): 
                     allValues2[k] = 1 / float(allValues[k])
                nextIndex = np.random.choice(list(allValues.keys()), 1, p = [ x/sum(allValues2.values()) for x in allValues2.values()])[0]

            if self.keypoint[index] in self.usedActionsID.keys() and self.keypoint[index] != 0 : 
                self.usedActionsID[self.keypoint[index]].append(self.indexToActionID[index])
            elif self.keypoint[index] != 0 : 
                self.usedActionsID[self.keypoint[index]] = [self.indexToActionID[index]]
            return self.treeTraversal(nextIndex)
            
        else: 
            nprint("tree traversal terminated on node %s"%(index))
            availableActions = copy.deepcopy(self.actions)
            #for k in self.usedActionsID.keys(): 
            #    for i in self.usedActionsID[k]: 
            #        availableActions[k].pop(i, None)
            return (index,availableActions)
        
    def initialiseExplorationNode(self,index,availableActions):
        nprint("expanding %s"%(index))
        if self.keypoint[index] != 0: 
            for (actionId, am) in availableActions[self.keypoint[index]].items() : 
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = 0 
                self.indexToActionID[self.indexToNow] = actionId
                self.initialiseLeafNode(self.indexToNow,index,am)
                self.children[index].append(self.indexToNow)
        else: 
            for kp in self.keypoints.keys() : 
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = kp
                self.indexToActionID[self.indexToNow] = 0
                self.initialiseLeafNode(self.indexToNow,index,{})
                self.children[index].append(self.indexToNow) 
        self.fullyExpanded[index] = True
        self.usedActionsID = {}
        return self.children[index]

    def backPropagation(self,index,value): 
        self.cost[index] += value
        self.numberOfVisited[index] += 1
        if self.parent[index] in self.parent : 
            nprint("start backPropagating the value %s from node %s, whose parent node is %s"%(value,index,self.parent[index]))
            self.backPropagation(self.parent[index],value)
        else: 
            nprint("backPropagating ends on node %s"%(index))
                    
    # start random sampling and return the Euclidean value as the value
    def sampling(self,index,availableActions):
        nprint("start sampling node %s"%(index))
        availableActions2 = copy.deepcopy(availableActions)
        #availableActions2[self.keypoint[index]].pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples): 
            self.atomicManipulationPath = self.manipulation[index]
            self.depth = 0
            self.availableActionIDs = {}
            for k in self.keypoints.keys(): 
                self.availableActionIDs[k] = list(availableActions2[k].keys())
            self.usedActionIDs = {}
            for k in self.keypoints.keys(): 
                self.usedActionIDs[k] = []             
            (childTerminated, val) = self.sampleNext(self.keypoint[index])
            sampleValues.append(val)
            i += 1
        return (childTerminated, min(sampleValues))
    
    def sampleNext(self,k): 
        
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulationPath)
        (newClass,newConfident) = self.model.predict(activations1)
        (distMethod,distVal) = self.eta
        if distMethod == "L2": 
            dist = l2Distance(activations1,self.image) 
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.image) 
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.image)
        elif distMethod == "NumDiffs": 
            dist =  diffPercent(activations1,self.image) * self.image.size

        # need not only class change, but also high confidence adversary examples
        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
            nprint("sampling a path ends in a terminal node with depth %s... "%self.depth)
            self.atomicManipulationPath = self.scrutinizePath(self.atomicManipulationPath)
            self.numAdv += 1
            nprint("current best %s, considered to be replaced by %s"%(self.bestCase[0],dist))
            if self.bestCase[0] > dist: 
                print("update best case from %s to %s"%(self.bestCase[0], dist))
                self.numConverge += 1
                self.bestCase = (dist,self.atomicManipulationPath)
                path0="%s_pic/%s_currentBest_%s.png"%(self.data_set,self.image_index,self.numConverge)
                self.model.saveInput(activations1,path0)
                if self.gameType == 'competitive':
                    print("the number of max features is: %s"%len(self.bestFeatures()[0]))
            return (self.depth == 0, dist)
            
        elif dist > distVal: 
            nprint("sampling a path ends by eta with depth %s ... "%self.depth)
            return (self.depth == 0, distVal)
            
        elif list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k])) == []: 
            nprint("sampling a path ends with depth %s because no more actions can be taken ... "%self.depth)
            return (self.depth == 0, distVal)
            
        else: 
            #print("continue sampling node ... ")
            #randomActionIndex = random.choice(list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k]))) 
            randomActionIndex = random.choice(self.availableActionIDs[k]) 
            if k == 0: 
                nextAtomicManipulation = {}
            else: 
                nextAtomicManipulation = self.actions[k][randomActionIndex]
                #self.availableActionIDs[k].remove(randomActionIndex)
                #self.usedActionIDs[k].append(randomActionIndex)
            newManipulationPath = mergeTwoDicts(self.atomicManipulationPath,nextAtomicManipulation)
            activations2 = self.moves.applyManipulation(self.image,newManipulationPath)
            (newClass2,newConfident2) = self.model.predict(activations2)

            self.atomicManipulationPath = newManipulationPath
            self.depth = self.depth+1
            if k == 0: 
                return self.sampleNext(randomActionIndex)
            else: 
                return self.sampleNext(0)
            
    def scrutinizePath(self, manipulations): 
        flag = False
        tempManipulations = copy.deepcopy(manipulations)
        for k, v in manipulations.items(): 
            tempManipulations[k] = 0
            activations1 = self.moves.applyManipulation(self.image,tempManipulations)
            (newClass,newConfident) = self.model.predict(activations1)
            if newClass != self.originalClass:
                manipulations.pop(k)
                flag = True
                break
            
        if flag == True: 
            return self.scrutinizePath(manipulations)
        else: 
            return manipulations
            
    def terminalNode(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        (newClass,_) = self.model.predict(activations1)
        return newClass != self.originalClass 
        
    def terminatedByEta(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        (distMethod,distVal) = self.eta
        if distMethod == "L2": 
            dist = l2Distance(activations1,self.image) 
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.image) 
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.image)
        elif distMethod == "NumDiffs": 
            dist = diffPercent(activations1,self.image)
        nprint("terminated by controlled search: distance = %s"%(dist))
        return dist > distVal 
        
    def applyManipulationToGetImage(self,manipulation):
        activations1 = self.moves.applyManipulation(self.image,manipulation)
        return activations1
        
    def l2Dist(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        return l2Distance(self.image,activations1)
        
    def l1Dist(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        return l1Distance(self.image,activations1)
        
    def l0Dist(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        return l0Distance(self.image,activations1)
        
    def diffImage(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        return diffImage(self.image,activations1)
        
    def diffPercent(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.manipulation[index])
        return diffPercent(self.image,activations1)
    
    def bestFeatures(self):
        bestManipulation = self.bestCase[1] 
        maxdims = []
        nf = 0 
        for i in range(1,len(self.actions)):
            md = []
            flag = False 
            for k, v in bestManipulation.items(): 
                for k1, v1 in self.actions[i].items(): 
                    md += list(v1.keys())
                    if k in v1.keys(): 
                        flag = True
            if flag == True: 
                nf += 1
                maxdims += md 
        if len(maxdims[0]) == 3: 
            maxdims = zip (*maxdims)
            maxdims = zip(list(maxdims)[0], list(maxdims)[1])
        return (nf,maxdims)
        