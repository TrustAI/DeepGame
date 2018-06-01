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
    (label, confident) = NN.predict(image)
    orig_label_str = NN.get_label(int(label))
    print("Working on input with index %s, whose class is '%s' and the confidence is %s."
          % (image_index, orig_label_str, confident))
    print("the second player is %s." % game_type)

    if game_type == 'cooperative':
        coop_game = CooperativeAStar(image, NN, eta, tau)
        coop_game.player1(image)

    elif game_type == 'competitive':
        comp_game = CompetitiveAlphaBeta(image, NN, eta, tau)

    else:
        print("Unrecognised game type. Try 'cooperative' or 'competitive'.")
