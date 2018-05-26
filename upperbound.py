from __future__ import print_function
from neural_network import *
from data_set import *
from mcts import *

def upperbound(dataSetName,bound,gameType,image_index,eta):

    MCTS_all_maximal_time = 300 
    MCTS_level_maximal_time = 60

    NN = NeuralNetwork(dataSetName)
    NN.load_network()
    print('Dataset is', NN.data_set)
    NN.model.summary()

    dataset = DataSet(dataSetName,'testing')
    image = dataset.getInput(image_index)
    (label,confident) = NN.predict(image)
    print("Working on input with index %s, whose class is %s and the confidence is %s."%(image_index,label,confident))

    tau = 1
    # choose between "cooperative" and "competitive "
    mctsInstance = mcts(dataSetName, NN, image_index, image, gameType, tau, eta)
    mctsInstance.initialiseMoves()

    start_time_all = time.time()
    runningTime_all = 0
    numberOfMoves = 0
    while mctsInstance.terminalNode(mctsInstance.rootIndex) == False and mctsInstance.terminatedByEta(mctsInstance.rootIndex) == False and runningTime_all <= MCTS_all_maximal_time: 
        print("the number of moves we have made up to now: %s"%(numberOfMoves))
        eudist = mctsInstance.l2Dist(mctsInstance.rootIndex)
        l1dist = mctsInstance.l1Dist(mctsInstance.rootIndex)
        l0dist = mctsInstance.l0Dist(mctsInstance.rootIndex)
        percent = mctsInstance.diffPercent(mctsInstance.rootIndex)
        diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
        print("L2 distance %s"%(eudist))
        print("L1 distance %s"%(l1dist))
        print("L0 distance %s"%(l0dist))
        print("manipulated percentage distance %s"%(percent))
        print("manipulated dimensions %s"%(diffs))

        start_time_level = time.time()
        runningTime_level = 0
        childTerminated = False
        while runningTime_level <= MCTS_level_maximal_time: 
            # Here are three steps for MCTS
            (leafNode,availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
            newNodes = mctsInstance.initialiseExplorationNode(leafNode,availableActions)
            for node in newNodes: 
                (childTerminated, value) = mctsInstance.sampling(node,availableActions)
                mctsInstance.backPropagation(node,value)
            runningTime_level = time.time() - start_time_level   
            print("best possible disance up to now is %s"%(str(mctsInstance.bestCase[0])))
        bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
        # pick the current best move to take  
        mctsInstance.makeOneMove(bestChild)
                
        image1 = mctsInstance.applyManipulationToGetImage(mctsInstance.manipulation[mctsInstance.rootIndex])
        diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
        path0="%s_pic/%s_temp_%s.png"%(dataSetName,image_index,len(diffs))
        NN.saveInput(image1,path0)
        (newClass,newConfident) = NN.predict(image1)
        print("confidence: %s"%(newConfident))
                
        # break if we found that one of the children is a misclassification
        if childTerminated == True: break
                
        # store the current best
        (_,bestManipulation) = mctsInstance.bestCase
        image1 = mctsInstance.applyManipulationToGetImage(bestManipulation)
        path0="%s_pic/%s_currentBest.png"%(dataSetName,image_index)
        NN.saveInput(image1,path0)
                
        numberOfMoves += 1
        runningTime_all = time.time() - start_time_all  