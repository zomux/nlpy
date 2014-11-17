#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest

import theano.tensor as T
import numpy as np
from nlpy.deep import NeuralRegressor

from nlpy.deep.networks.basic_nn import NeuralNetwork
from nlpy.deep.trainers.trainer import SGDTrainer


class FeedForwardTest(unittest.TestCase):

    def test(self):
        from nlpy.dataset import HeartScaleDataset
        from nlpy.deep.conf import NetworkConfig
        from nlpy.deep import NeuralLayer
        import logging
        logging.basicConfig(level=logging.INFO)
        conf = NetworkConfig(input_size=13)
        conf.layers = [NeuralLayer(10), NeuralLayer(5), NeuralLayer(1, 'linear')]
        ff = NeuralRegressor(conf)
        t = SGDTrainer(ff)
        train_set = [(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13]]), np.array([[1,0]]))]
        a = [HeartScaleDataset(single_target=True).train_set()]
        b = [HeartScaleDataset(single_target=True).valid_set()]
        for k in list(t.train(a, b)):
            pass
        print k


if __name__ == '__main__':
    unittest.main()