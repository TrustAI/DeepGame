#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Construct a CompetitiveAlphaBeta class to compute
the lower bound of Player I’s maximum adversary distance
while Player II being competitive.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""


class CompetitiveAlphaBeta:
    def __init__(self, image, model, eta, tau):
        self.IMAGE = image
        self.MODEL = model
        self.ETA = eta
        self.TAU = tau
