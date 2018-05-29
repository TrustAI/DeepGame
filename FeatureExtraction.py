"""
Construct a FeatureExtraction class to retrieve
'key points' and 'partitions' of an image
in a black-box or grey-box pattern.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import copy
import numpy as np
import cv2
from scipy.stats import norm
import random
from keras import backend as K

IMAGE_SIZE_BOUND = 100
MAX_NUM_OF_PIXELS_PER_KEY_POINT = 1000000
NUM_OF_PIXEL_MANIPULATION = 2


class FeatureExtraction:
    def __init__(self, image, pattern='black-box', model=None):
        self.IMAGE = image
        self.MODEL = model
        self.PATTERN = pattern

        if max(self.IMAGE.shape) < IMAGE_SIZE_BOUND:
            self.IMAGE_SIZE = 'small'
        else:
            self.IMAGE_SIZE = 'large'

        self.img_enlarge_ratio = 1

        if self.PATTERN == 'grey-box' and self.MODEL is None:
            print("For 'grey-box' feature extraction, please specify a neural network.")
            exit

    def get_partitions(self, pixel_bounds=(0, 1), num_partition=10):
        if self.PATTERN == 'black-box':
            print("Extracting image features using '%s' pattern." % self.PATTERN)

            key_points = self.get_key_points()
            print("%s keypoints are found. " % (len(key_points)))

            partitions = {}
            if self.IMAGE_SIZE == 'small':
                for x in range(max(self.IMAGE.shape)):
                    for y in range(max(self.IMAGE.shape)):
                        ps = 0
                        maxk = -1
                        for i in range(1, len(key_points) + 1):
                            k = key_points[i - 1]
                            dist2 = np.linalg.norm(np.array([x, y]) - np.array([k.pt[0], k.pt[1]]))
                            ps2 = norm.pdf(dist2, loc=0.0, scale=k.size)
                            if ps2 > ps:
                                ps = ps2
                                maxk = i
                        if maxk in partitions.keys():
                            partitions[maxk].append((x, y))
                        else:
                            partitions[maxk] = [(x, y)]
                if MAX_NUM_OF_PIXELS_PER_KEY_POINT > 0:
                    for mk in partitions.keys():
                        begining_num = len(partitions[mk])
                        for i in range(begining_num - MAX_NUM_OF_PIXELS_PER_KEY_POINT):
                            partitions[mk].remove(random.choice(partitions[mk]))
                return partitions
            else:
                key_points = key_points[:200]
                each_num = max(self.IMAGE.shape) ** 2 / len(key_points)
                maxk = 1
                partitions[maxk] = []
                for x in range(max(self.IMAGE.shape)):
                    for y in range(max(self.IMAGE.shape)):
                        if len(partitions[maxk]) <= each_num:
                            partitions[maxk].append((x, y))
                        else:
                            maxk += 1
                            partitions[maxk] = [(x, y)]
                return partitions

        elif self.PATTERN == 'grey-box':
            print("Extracting image features using '%s' pattern." % self.PATTERN)

            image_class, _ = self.MODEL.predict(self.IMAGE)

            new_pixel_list = np.linspace(pixel_bounds[0], pixel_bounds[1], NUM_OF_PIXEL_MANIPULATION)
            image_batch = np.kron(np.ones((NUM_OF_PIXEL_MANIPULATION, 1, 1, 1)), self.IMAGE)

            manipulated_images = []
            (row, col, chl) = self.IMAGE.shape
            for i in range(0, row):
                for j in range(0, col):
                    # need to be very careful about image.copy()
                    changed_image_batch = image_batch.copy()
                    for p in range(0, NUM_OF_PIXEL_MANIPULATION):
                        changed_image_batch[p, i, j, :] = new_pixel_list[p]
                    manipulated_images.append(changed_image_batch)  # each loop append [pixel_num, row, col, chl]

            manipulated_images = np.asarray(manipulated_images)  # [row*col, pixel_num, row, col, chl]
            manipulated_images = manipulated_images.reshape(row * col * NUM_OF_PIXEL_MANIPULATION, row, col, chl)

            features_list = self.softmax_logits(manipulated_images, self.MODEL.model)
            feature_change = features_list[:, image_class].reshape(-1, NUM_OF_PIXEL_MANIPULATION).transpose()

            min_indices = np.argmin(feature_change, axis=0)
            min_values = np.amin(feature_change, axis=0)
            min_idx_values = min_indices.astype('float32') / (NUM_OF_PIXEL_MANIPULATION - 1)

            [x, y] = np.meshgrid(np.arange(row), np.arange(col))
            x = x.flatten('F')  # to flatten in column-major order
            y = y.flatten('F')  # to flatten in column-major order

            target_feature_list = np.hstack((np.split(x, len(x)),
                                             np.split(y, len(y)),
                                             np.split(min_values, len(min_values)),
                                             np.split(min_idx_values, len(min_idx_values))))

            saliency_map = target_feature_list[target_feature_list[:, 2].argsort()]

            partitions = {}
            quotient, remainder = divmod(len(saliency_map), num_partition)
            for key in range(1, num_partition + 1):
                partitions[key] = [(int(saliency_map[idx, 0]), int(saliency_map[idx, 1])) for idx in
                                   range((key - 1) * quotient, key * quotient)]
                if key == num_partition:
                    partitions[key].extend((int(saliency_map[idx, 0]), int(saliency_map[idx, 1])) for idx in
                                           range(key * quotient, len(saliency_map)))

            return partitions

        else:
            print("Unrecognised feature extraction pattern. "
                  "Try 'black-box' or 'grey-box'.")

    def get_key_points(self, num_partition=10):
        if self.PATTERN == 'black-box':
            image = copy.deepcopy(self.IMAGE)

            sift = cv2.xfeatures2d.SIFT_create()  # cv2.SIFT() # cv2.SURF(400)

            if np.max(image) <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

            if self.IMAGE_SIZE == 'small':
                # for a small image, SIFT works by enlarging the image
                image = cv2.resize(image, (0, 0), fx=self.img_enlarge_ratio, fy=self.img_enlarge_ratio)
                key_points, _ = sift.detectAndCompute(image, None)
                for i in range(len(key_points)):
                    old_pt = (key_points[i].pt[0], key_points[i].pt[1])
                    key_points[i].pt = (int(old_pt[0] / self.img_enlarge_ratio),
                                        int(old_pt[1] / self.img_enlarge_ratio))
            else:
                key_points, _ = sift.detectAndCompute(image, None)

        elif self.PATTERN == 'grey-box':
            key_points = [key for key in range(num_partition)]
            # key_points = {}
            # for key in range(num_partition):
            #     key_points[key] = []

        return key_points

    def softmax_logits(self, manipulated_images, model):
        # get logits of softmax function, as softmax probabilities
        # may be too close to each other after just one pixel manipulation

        func = K.function([model.layers[0].input] + [K.learning_phase()],
                          [model.layers[model.layers.__len__() - 1].output.op.inputs[0]])

        # func = K.function([model.layers[0].input] + [K.learning_phase()],
        #                   [model.layers[model.layers.__len__() - 1].output])

        softmax_logits = func([manipulated_images, 0])[0]
        return softmax_logits
