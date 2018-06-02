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
from DateSet import *


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
    print("the second player is being %s." % game_type)

    path = "%s_pic/%s_%s_with_confidence_%s.png" % (
        dataset_name, image_index, label_str, confidence)
    NN.save_input(image, path)

    if game_type == 'cooperative':
        cooperative = CooperativeAStar(image, NN, eta, tau)
        cooperative.play_game(image)
        if cooperative.ADVERSARY_FOUND is True:
            adversary = cooperative.ADVERSARY
            adv_label, adv_confidence = NN.predict(adversary)
            adv_label_str = NN.get_label(int(adv_label))

            print("\nfound an adversary image within pre-specified bounded computational resource. "
                  "The following is its information: ")
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

            path = "%s_pic/%s_%s_modified_into_%s_with_confidence_%s.png" % (
                dataset_name, image_index, label_str, adv_label_str, adv_confidence)
            NN.save_input(adversary, path)
            path = "%s_pic/%s_diff_L2_%s_L1_%s_L0_%s.png" % (
                dataset_name, image_index, l2dist, l1dist, l0dist)
            NN.save_input(np.subtract(image, adversary), path)
        else:
            print("Adversarial distance exceeds distance bound.")

    elif game_type == 'competitive':
        competitive = CompetitiveAlphaBeta(image, NN, eta, tau)

    else:
        print("Unrecognised game type. Try 'cooperative' or 'competitive'.")
