from __future__ import print_function
from neural_network import *
from DateSet import *
from MCTSCompetitive import *
from MCTSCooperative import *


def upperbound(dataSetName, bound, tau, gameType, image_index, eta):
    start_time = time.time()

    MCTS_all_maximal_time = 300
    MCTS_level_maximal_time = 60

    NN = NeuralNetwork(dataSetName)
    NN.load_network()
    print("Dataset is %s." % NN.data_set)
    NN.model.summary()

    dataset = DataSet(dataSetName, 'testing')
    image = dataset.get_input(image_index)
    (label, confident) = NN.predict(image)
    origClassStr = NN.get_label(int(label))
    print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, origClassStr, confident))

    tau = 1
    # choose between "cooperative" and "competitive"
    if gameType == 'cooperative':
        mctsInstance = MCTSCooperative(dataSetName, NN, image_index, image, tau, eta)
        mctsInstance.initialiseMoves()

        start_time_all = time.time()
        runningTime_all = 0
        numberOfMoves = 0
        while (not mctsInstance.terminalNode(mctsInstance.rootIndex) and
               not mctsInstance.terminatedByEta(mctsInstance.rootIndex) and
               runningTime_all <= MCTS_all_maximal_time):
            print("the number of moves we have made up to now: %s" % numberOfMoves)
            l2dist = mctsInstance.l2Dist(mctsInstance.rootIndex)
            l1dist = mctsInstance.l1Dist(mctsInstance.rootIndex)
            l0dist = mctsInstance.l0Dist(mctsInstance.rootIndex)
            percent = mctsInstance.diffPercent(mctsInstance.rootIndex)
            diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("manipulated dimensions %s" % diffs)

            start_time_level = time.time()
            runningTime_level = 0
            childTerminated = False
            while runningTime_level <= MCTS_level_maximal_time:
                # Here are three steps for MCTS
                (leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
                newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
                for node in newNodes:
                    (childTerminated, value) = mctsInstance.sampling(node, availableActions)
                    mctsInstance.backPropagation(node, value)
                runningTime_level = time.time() - start_time_level
                print("best possible distance up to now is %s" % (str(mctsInstance.bestCase[0])))
            bestChild = mctsInstance.bestChild(mctsInstance.rootIndex)
            # pick the current best move to take  
            mctsInstance.makeOneMove(bestChild)

            image1 = mctsInstance.applyManipulation(mctsInstance.manipulation[mctsInstance.rootIndex])
            diffs = mctsInstance.diffImage(mctsInstance.rootIndex)
            path0 = "%s_pic/%s_temp_%s.png" % (dataSetName, image_index, len(diffs))
            NN.save_input(image1, path0)
            (newClass, newConfident) = NN.predict(image1)
            print("confidence: %s" % newConfident)

            # break if we found that one of the children is a misclassification
            if childTerminated is True:
                break

            # store the current best
            (_, bestManipulation) = mctsInstance.bestCase
            image1 = mctsInstance.applyManipulation(bestManipulation)
            path0 = "%s_pic/%s_currentBest.png" % (dataSetName, image_index)
            NN.save_input(image1, path0)

            numberOfMoves += 1
            runningTime_all = time.time() - start_time_all

        (_, bestManipulation) = mctsInstance.bestCase

        image1 = mctsInstance.applyManipulation(bestManipulation)
        (newClass, newConfident) = NN.predict(image1)
        newClassStr = NN.get_label(int(newClass))

        if newClass != label:
            path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
                dataSetName, image_index, origClassStr, newClassStr, newConfident)
            NN.save_input(image1, path0)
            path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
            NN.save_input(np.subtract(image, image1), path0)
            print("\nfound an adversary image within pre-specified bounded computational resource. "
                  "The following is its information: ")
            print("difference between images: %s" % (diffImage(image, image1)))

            print("number of adversarial examples found: %s" % mctsInstance.numAdv)

            l2dist = l2Distance(mctsInstance.image, image1)
            l1dist = l1Distance(mctsInstance.image, image1)
            l0dist = l0Distance(mctsInstance.image, image1)
            percent = diffPercent(mctsInstance.image, image1)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))

            return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, 0

        else:
            print("\nfailed to find an adversary image within pre-specified bounded computational resource. ")
            return 0, 0, 0, 0, 0, 0, 0
        
    elif gameType == 'competitive':
    
        mctsInstance = MCTSCompetitive(dataSetName, NN, image_index, image, tau, eta)
        mctsInstance.initialiseMoves()

        start_time_all = time.time()
        runningTime_all = 0
        while runningTime_all <= MCTS_all_maximal_time:

            (leafNode, availableActions) = mctsInstance.treeTraversal(mctsInstance.rootIndex)
            newNodes = mctsInstance.initialiseExplorationNode(leafNode, availableActions)
            for node in newNodes:
                (childTerminated, value) = mctsInstance.sampling(node, availableActions)
                mctsInstance.backPropagation(node, value)
            print("best possible distance up to now is %s" % (str(mctsInstance.bestCase[0])))

            # store the current best
            (_, bestManipulation) = mctsInstance.bestCase
            image1 = mctsInstance.applyManipulation(bestManipulation)
            path0 = "%s_pic/%s_currentBest.png" % (dataSetName, image_index)
            NN.save_input(image1, path0)

            runningTime_all = time.time() - start_time_all

        (bestValue, bestManipulation) = mctsInstance.bestCase

        print("the number of max features is %s" % mctsInstance.bestFeatures()[0])
        maxfeatures = mctsInstance.bestFeatures()[0]

        if bestValue < eta[1]: 

            image1 = mctsInstance.applyManipulation(bestManipulation)
            (newClass, newConfident) = NN.predict(image1)
            newClassStr = NN.get_label(int(newClass))

            if newClass != label:
                path0 = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
                    dataSetName, image_index, origClassStr, newClassStr, newConfident)
                NN.save_input(image1, path0)
                path0 = "%s_pic/%s_diff.png" % (dataSetName, image_index)
                NN.save_input(np.subtract(image, image1), path0)
                print("\nfound an adversary image within pre-specified bounded computational resource. "
                     "The following is its information: ")
                print("difference between images: %s" % (diffImage(image, image1)))

                print("number of adversarial examples found: %s" % mctsInstance.numAdv)

                l2dist = l2Distance(mctsInstance.image, image1)
                l1dist = l1Distance(mctsInstance.image, image1)
                l0dist = l0Distance(mctsInstance.image, image1)
                percent = diffPercent(mctsInstance.image, image1)
                print("L2 distance %s" % l2dist)
                print("L1 distance %s" % l1dist)
                print("L0 distance %s" % l0dist)
                print("manipulated percentage distance %s" % percent)
                print("class is changed into '%s' with confidence %s\n" % (newClassStr, newConfident))

                return time.time() - start_time_all, newConfident, percent, l2dist, l1dist, l0dist, maxfeatures

            else:
                print("\nthere exists a feature which, up to now, hasn't been discovered with an adversarial exammple. ")
                return 0, 0, 0, 0, 0, 0, 0
            
    else:
        print("Unrecognised game type. Try 'cooperative' or 'competitive'.")

    runningTime = time.time() - start_time
