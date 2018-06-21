#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a 'lowerbound' function to compute
the lower bound of Player I’s minimum adversary distance
while Player II being cooperative, or Player I's maximum
adversary distance whilst Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

from CooperativeAStar import *
from CompetitiveAlphaBeta import *
from NeuralNetwork import *
from DataSet import *


def lowerbound(dataset_name, image_index, game_type, eta, tau):
    NN = NeuralNetwork(dataset_name)
    NN.load_network()
    print("Dataset is %s." % NN.data_set)
    NN.model.summary()

    dataset = DataSet(dataset_name, 'testing')
    image = dataset.get_input(image_index)
    (label, confidence) = NN.predict(image)
    label_str = NN.get_label(int(label))
    print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, label_str, confidence))
    print("The second player is being %s." % game_type)

    path = "%s_pic/idx_%s_label_[%s]_with_confidence_%s.png" % (
        dataset_name, image_index, label_str, confidence)
    NN.save_input(image, path)

    if game_type == 'cooperative':
        tic = time.time()
        cooperative = CooperativeAStar(image_index, image, NN, eta, tau)
        cooperative.play_game(image)
        if cooperative.ADVERSARY_FOUND is True:
            elapsed = time.time() - tic
            adversary = cooperative.ADVERSARY
            adv_label, adv_confidence = NN.predict(adversary)
            adv_label_str = NN.get_label(int(adv_label))

            print("\nFound an adversary within pre-specified bounded computational resource. "
                  "\nThe following is its information: ")
            print("difference between images: %s" % (diffImage(image, adversary)))
            l2dist = l2Distance(image, adversary)
            l1dist = l1Distance(image, adversary)
            l0dist = l0Distance(image, adversary)
            percent = diffPercent(image, adversary)
            print("L2 distance %s" % l2dist)
            print("L1 distance %s" % l1dist)
            print("L0 distance %s" % l0dist)
            print("manipulated percentage distance %s" % percent)
            print("class is changed into '%s' with confidence %s\n" % (adv_label_str, adv_confidence))

            path = "%s_pic/idx_%s_modified_into_[%s]_with_confidence_%s.png" % (
                dataset_name, image_index, adv_label_str, adv_confidence)
            NN.save_input(adversary, path)
            if eta[0] == 'L0':
                dist = l0dist
            elif eta[0] == 'L1':
                dist = l1dist
            elif eta[0] == 'L2':
                dist = l2dist
            else:
                print("Unrecognised distance metric.")
            path = "%s_pic/idx_%s_modified_diff_%s=%s_time=%s.png" % (
                dataset_name, image_index, eta[0], dist, elapsed)
            NN.save_input(np.absolute(image - adversary), path)
        else:
            print("Adversarial distance exceeds distance bound.")

    elif game_type == 'competitive':
        competitive = CompetitiveAlphaBeta(image, NN, eta, tau)
        competitive.play_game(image)

    else:
        print("Unrecognised game type. Try 'cooperative' or 'competitive'.")
