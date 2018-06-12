#!/usr/bin/env python


import numpy as np
import math
import time
import os
import copy


def assure_path_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def current_milli_time():
    return int(round(time.time() * 1000) % 4294967296)


def diffImage(image1, image2):
    return list(zip(*np.nonzero(np.subtract(image1, image2))))


def diffPercent(image1, image2):
    return len(diffImage(image1, image2)) / float(image1.size)


def numDiffs(image1, image2):
    return len(diffImage(image1, image2))


def l2Distance(image1, image2):
    return math.sqrt(np.sum(np.square(np.subtract(image1, image2))))


def l1Distance(image1, image2):
    return np.sum(np.absolute(np.subtract(image1, image2)))


def l0Distance(image1, image2):
    return np.count_nonzero(np.absolute(np.subtract(image1, image2)))


def mergeTwoDicts(x, y):
    z = x.copy()
    for key, value in y.items():
        if key in z.keys():
            z[key] += y[key]
        else:
            z[key] = y[key]
    # z.update(y)
    return z


def nprint(str):
    return 0


def printDict(dictionary):
    for key, value in dictionary.items():
        print("%s : %s" % (key, value))
    print("\n")
