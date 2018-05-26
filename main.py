from __future__ import print_function
from neural_network import *
from data_set import *
from mcts import *

#dataSetName = 'mnist'
dataSetName = 'cifar10'


image_index = 3

gameType = "cooperator"  # "competitor"
eta = ("L1",40)
MCTS_all_maximal_time = 300 
MCTS_level_maximal_time = 60

NN = NeuralNetwork(dataSetName)
# NN.train_network()
# NN.save_network()
NN.load_network()
print('Dataset is', NN.data_set)
NN.model.summary()

dataset = DataSet(dataSetName,'testing')
image = dataset.getInput(image_index)
(label,confident) = NN.predict(image)
print("Working on input with index %s, whose class is %s and the confidence is %s."%(image_index,label,confident))

tau = 1
# choose between "cooperator" and "competitor"
mcts = mcts(dataSetName, NN, image_index, image, gameType, tau, eta)
mcts.initialiseMoves()

start_time_all = time.time()
runningTime_all = 0
numberOfMoves = 0
while mcts.terminalNode(mcts.rootIndex) == False and mcts.terminatedByEta(mcts.rootIndex) == False and runningTime_all <= MCTS_all_maximal_time: 
    print("the number of moves we have made up to now: %s"%(numberOfMoves))
    eudist = mcts.l2Dist(mcts.rootIndex)
    l1dist = mcts.l1Dist(mcts.rootIndex)
    l0dist = mcts.l0Dist(mcts.rootIndex)
    percent = mcts.diffPercent(mcts.rootIndex)
    diffs = mcts.diffImage(mcts.rootIndex)
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
        (leafNode,availableActions) = mcts.treeTraversal(mcts.rootIndex)
        newNodes = mcts.initialiseExplorationNode(leafNode,availableActions)
        for node in newNodes: 
            (childTerminated, value) = mcts.sampling(node,availableActions)
            mcts.backPropagation(node,value)
        runningTime_level = time.time() - start_time_level   
        print("best possible one is %s"%(str(mcts.bestCase)))
    bestChild = mcts.bestChild(mcts.rootIndex)
    # pick the current best move to take  
    mcts.makeOneMove(bestChild)
                
    image1 = mcts.applyManipulationToGetImage(mcts.manipulation[mcts.rootIndex])
    diffs = mcts.diffImage(mcts.rootIndex)
    path0="%s/%s_temp_%s.png"%(directory_pic_string,startIndexOfImage,len(diffs))
    dataBasics.save(-1,image1,path0)
    (newClass,newConfident) = NN.predictWithImage(model,image1)
    print("confidence: %s"%(newConfident))
                
    if childTerminated == True: break
                
    # store the current best
    (_,bestatomicManipulation,bestnumberAtomicManipulation) = mcts.bestCase
    image1 = mcts.applyManipulationToGetImage(bestatomicManipulation,bestnumberAtomicManipulation)
    path0="%s/%s_currentBemcts.png"%(directory_pic_string,startIndexOfImage)
    dataBasics.save(-1,image1,path0)
                
    numberOfMoves += 1
    runningTime_all = time.time() - start_time_all  