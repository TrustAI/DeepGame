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


MCTS_multi_samples = 3
effectiveConfidenceWhenChanging = 0.8
explorationRate = math.sqrt(2)        



class mcts:

    def __init__(self, data_set, model, image_index, image, player_mode, tau, eta):
        self.data_set = data_set
        self.image_index = image_index
        self.image = image
        self.model = model
        self.player_mode = player_mode
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
                
        self.atomicManipulation = {}
        self.numberAtomicManipulation = {}
        # initialise root node
        self.atomicManipulation[-1] = {}
        self.numberAtomicManipulation[-1] = {}
        self.initialiseLeafNode(0,-1,[],[])
        
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
        self.bestCase = (0,{},{})
        self.numConverge = 0 
        
        # number of adversarial exmaples
        self.numAdv = 0
        #self.analyseAdv = analyseAdv(activations)

                
        # temporary variables for sampling 
        self.atomicManipulationPath = []
        self.numberAtomicManipulationPath = [] 
        self.depth = 0
        self.availableActionIDs = []
        self.usedActionIDs = [] 
        self.accDims = [] 
        self.d =0
        
        
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
        
    def initialiseLeafNode(self,index,parentIndex,newatomicManipulation,newnumberAtomicManipulation):
        nprint("initialising a leaf node %s from the node %s"%(index,parentIndex))
        self.atomicManipulation[index] = mergeTwoDicts(self.atomicManipulation[parentIndex],newatomicManipulation)
        self.numberAtomicManipulation[index] = mergeTwoDicts(self.numberAtomicManipulation[parentIndex],newnumberAtomicManipulation)
        self.cost[index] = 0
        self.parent[index] = parentIndex 
        self.children[index] = []
        self.fullyExpanded[index] = False
        self.numberOfVisited[index] = 0    
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])


    def destructor(self): 
        self.image = 0
        self.image = 0
        self.model = 0
        self.model = 0
        self.atomicManipulation = {}
        self.numberAtomicManipulation = {}
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
        print("making a move into the new root %s, whose value is %s and visited number is %s"%(newRootIndex,self.cost[newRootIndex],self.numberOfVisited[newRootIndex]))
        self.removeChildren(self.rootIndex,[newRootIndex])
        self.rootIndex = newRootIndex
    
    def removeChildren(self,index,indicesToAvoid): 
        if self.fullyExpanded[index] == True: 
            for childIndex in self.children[index]: 
                if childIndex not in indicesToAvoid: self.removeChildren(childIndex,[])
        self.atomicManipulation.pop(index,None)
        self.numberAtomicManipulation.pop(index,None)
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
        return max(allValues.items(), key=operator.itemgetter(1))[0]
        
    def treeTraversal(self,index):
        if self.fullyExpanded[index] == True: 
            nprint("tree traversal on node %s"%(index))
            allValues = {}
            for childIndex in self.children[index]: 
                allValues[childIndex] = (self.cost[childIndex] / float(self.numberOfVisited[childIndex])) + explorationRate * math.sqrt(math.log(self.numberOfVisited[index]) / float(self.numberOfVisited[childIndex]))
            #nextIndex = max(allValues.iteritems(), key=operator.itemgetter(1))[0]
            if self.player_mode == "adversary" and self.keypoint[index] == 0 : 
                allValues2 = {}
                for k,v in allValues.items(): 
                     allValues2[k] = 1 / float(allValues[k])
                nextIndex = np.random.choice(list(allValues.keys()), 1, p = [ x/sum(allValues2.values()) for x in allValues2.values()])[0]
            else: 
                nextIndex = np.random.choice(list(allValues.keys()), 1, p = [ x/sum(allValues.values()) for x in allValues.values()])[0]

            if self.keypoint[index] in self.usedActionsID.keys() and self.keypoint[index] != 0 : 
                self.usedActionsID[self.keypoint[index]].append(self.indexToActionID[index])
            elif self.keypoint[index] != 0 : 
                self.usedActionsID[self.keypoint[index]] = [self.indexToActionID[index]]
            return self.treeTraversal(nextIndex)
        else: 
            nprint("tree traversal terminated on node %s"%(index))
            availableActions = copy.deepcopy(self.actions)
            for k in self.usedActionsID.keys(): 
                for i in self.usedActionsID[k]: 
                    availableActions[k].pop(i, None)
            return (index,availableActions)
        
    def initialiseExplorationNode(self,index,availableActions):
        nprint("expanding %s"%(index))
        if self.keypoint[index] != 0: 
            for (actionId, (span,numSpan,_)) in availableActions[self.keypoint[index]].items() : #initialisePixelSets(self.model,self.image,list(set(self.atomicManipulation[index].keys() + self.usefulPixels))): 
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = 0 
                self.indexToActionID[self.indexToNow] = actionId
                self.initialiseLeafNode(self.indexToNow,index,span,numSpan)
                self.children[index].append(self.indexToNow)
        else: 
            for kp in self.keypoints.keys() : #initialisePixelSets(self.model,self.image,list(set(self.atomicManipulation[index].keys() + self.usefulPixels))): 
                self.indexToNow += 1
                self.keypoint[self.indexToNow] = kp
                self.indexToActionID[self.indexToNow] = 0
                self.initialiseLeafNode(self.indexToNow,index,{},{})
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
        #print(availableActions,self.keypoint[index],self.indexToActionID[index])
        availableActions2[self.keypoint[index]].pop(self.indexToActionID[index], None)
        sampleValues = []
        i = 0
        for i in range(MCTS_multi_samples): 
            self.atomicManipulationPath = self.atomicManipulation[index]
            self.numberAtomicManipulationPath = self.numberAtomicManipulation[index] 
            self.depth = 0
            self.availableActionIDs = {}
            for k in self.keypoints.keys(): 
                self.availableActionIDs[k] = list(availableActions2[k].keys())
            self.usedActionIDs = {}
            for k in self.keypoints.keys(): 
                self.usedActionIDs[k] = []             
            self.accDims = [] 
            self.d = 2
            (childTerminated, val) = self.sampleNext(self.keypoint[index])
            sampleValues.append(val)
            #if childTerminated == True: break
            i += 1
        return (childTerminated, max(sampleValues))
    
    def sampleNext(self,k): 
        #print("k=%s"%k)
        #for j in self.keypoints: 
        #    print(len(self.availableActionIDs[j]))
        #print("oooooooo")
        
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulationPath,self.numberAtomicManipulationPath)
        (newClass,newConfident) = self.model.predict(activations1)
        (distMethod,distVal) = self.eta
        if distMethod == "euclidean": 
            dist = l2Distance(activations1,self.image) 
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.image) 
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.image)
            termValue = 0.0
            termByDist = dist > distVal
        elif distMethod == "NumDiffs": 
            dist =  diffPercent(activations1,self.image) * self.image.size
            termValue = 0.0
            termByDist = dist > distVal

        #if termByDist == False and newConfident < 0.5 and self.depth <= 3: 
        #    termByDist = True
        


        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
            # and newClass == self.model.next_index(self.originalClass,self.originalClass): 
            nprint("sampling a path ends in a terminal node with self.depth %s... "%self.depth)
            
            #print("L1 distance: %s"%(l1Distance(self.image,activations1)))
            #print(self.image.shape)
            #print(activations1.shape)
            #print("L1 distance with KL: %s"%(withKL(l1Distance(self.image,activations1),self.image,activations1)))
            
            (self.atomicManipulationPath,self.numberAtomicManipulationPath) = self.scrutinizePath(self.atomicManipulationPath,self.numberAtomicManipulationPath,newClass)
            
            #self.decisionTree.addOnePath(dist,self.atomicManipulationPath,self.numberAtomicManipulationPath)
            self.numAdv += 1

                
            if self.bestCase[0] < dist: 
                self.numConverge += 1
                self.bestCase = (dist,self.atomicManipulationPath,self.numberAtomicManipulationPath)
                path0="%s_pic/%s_currentBest_%s.png"%(self.data_set,self.image_index,self.numConverge)
                self.model.saveInput(activations1,path0)

            return (self.depth == 0, dist)
        elif termByDist == True: 
            nprint("sampling a path ends by controlled search with self.depth %s ... "%self.depth)
            return (self.depth == 0, termValue)
        elif list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k])) == []: 
            nprint("sampling a path ends with self.depth %s because no more actions can be taken ... "%self.depth)
            return (self.depth == 0, termValue)
        else: 
            #print("continue sampling node ... ")
            #allChildren = initialisePixelSets(self.model,self.image,self.atomicManipulationPath.keys())
            randomActionIndex = random.choice(list(set(self.availableActionIDs[k])-set(self.usedActionIDs[k]))) #random.randint(0, len(allChildren)-1)
            if k == 0: 
                span = {}
                numSpan = {}
            else: 
                (span,numSpan,_) = self.actions[k][randomActionIndex]
                self.availableActionIDs[k].remove(randomActionIndex)
                self.usedActionIDs[k].append(randomActionIndex)
            newSpanPath = self.mergeSpan(self.atomicManipulationPath,span)
            newNumSpanPath = self.mergeNumSpan(self.numberAtomicManipulationPath,numSpan)
            activations2 = self.moves.applyManipulation(self.image,newSpanPath,newNumSpanPath)
            (newClass2,newConfident2) = self.model.predict(activations2)
            confGap2 = newConfident - newConfident2
            if newClass2 == newClass: 
                self.accDims.append((randomActionIndex,confGap2))
            else: self.accDims.append((randomActionIndex,1.0))

            self.atomicManipulationPath = newSpanPath
            self.numberAtomicManipulationPath = newNumSpanPath
            self.depth = self.depth+1
            self.accDims = self.accDims
            self.d = self.d
            if k == 0: 
                return self.sampleNext(randomActionIndex)
            else: 
                return self.sampleNext(0)
            
    def scrutinizePath(self,spanPath,numSpanPath,changedClass): 
        lastSpanPath = copy.deepcopy(spanPath)
        for i in self.actions.keys(): 
            if i != 0: 
                for key, (span,numSpan,_) in self.actions[i].items(): 
                    if set(span.keys()).issubset(set(spanPath.keys())): 
                        tempSpanPath = copy.deepcopy(spanPath)
                        tempNumSpanPath = copy.deepcopy(numSpanPath)
                        for k in span.keys(): 
                            tempSpanPath.pop(k)
                            tempNumSpanPath.pop(k) 
                        activations1 = self.moves.applyManipulation(self.image,tempSpanPath,tempNumSpanPath)
                        (newClass,newConfident) = self.model.predict(activations1)
                        #if changedClass == newClass: 
                        if newClass != self.originalClass and newConfident > effectiveConfidenceWhenChanging:
                            for k in span.keys(): 
                                spanPath.pop(k)
                                numSpanPath.pop(k)
        if len(lastSpanPath.keys()) != len(spanPath.keys()): 
            return self.scrutinizePath(spanPath,numSpanPath,changedClass)
        else: 
            return (spanPath,numSpanPath)
            
    def terminalNode(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        (newClass,_) = self.model.predict(activations1)
        return newClass != self.originalClass 
        
    def terminatedByEta(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        (distMethod,distVal) = self.eta
        if distMethod == "euclidean": 
            dist = l2Distance(activations1,self.image) 
        elif distMethod == "L1": 
            dist = l1Distance(activations1,self.image) 
        elif distMethod == "Percentage": 
            dist = diffPercent(activations1,self.image)
        elif distMethod == "NumDiffs": 
            dist = diffPercent(activations1,self.image)
        nprint("terminated by controlled search: distance = %s"%(dist))
        return dist > distVal 
        
    def applyManipulationToGetImage(self,atomicManipulation,numberAtomicManipulation):
        activations1 = self.moves.applyManipulation(self.image,atomicManipulation,numberAtomicManipulation)
        return activations1
        
    def euclideanDist(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        return l2Distance(self.image,activations1)
        
    def l1Dist(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        return l1Distance(self.image,activations1)
        
    def l0Dist(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        return l0Distance(self.image,activations1)
        
    def diffImage(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        return diffImage(self.image,activations1)
        
    def diffPercent(self,index): 
        activations1 = self.moves.applyManipulation(self.image,self.atomicManipulation[index],self.numberAtomicManipulation[index])
        return diffPercent(self.image,activations1)

    def mergeSpan(self,atomicManipulationPath,span): 
        return mergeTwoDicts(atomicManipulationPath, span)
        
    def mergeNumSpan(self,numberAtomicManipulationPath,numSpan):
        return mergeTwoDicts(numberAtomicManipulationPath, numSpan)
        
    def showDecisionTree(self):
        self.decisionTree.show()
        self.decisionTree.outputTree()
    
        