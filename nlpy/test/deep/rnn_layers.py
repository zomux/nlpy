#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import theano
import theano.tensor as T
from nlpy.deep.networks.rnn_layers import RecurrentLayers
from nlpy.deep.conf import NetworkConfig, TrainerConfig
from nlpy.deep.functions import FLOATX
from nlpy.deep import NeuralLayer, SGDTrainer, NeuralClassifier
import time
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    net_conf = NetworkConfig(input_size=6)
    net_conf.layers = [RecurrentLayers(size=3, depth=3, activation='relu'), NeuralLayer(3, 'softmax')]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.03
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1000

    network = NeuralClassifier(net_conf)
    trainer = SGDTrainer(network)

    data = np.array([[1,0,0,0,0,0],
                     [0,1,0,0,0,0],
                     [0,0,1,0,0,0],
                     [0,0,0,1,0,0],
                     [0,0,0,0,1,0],
                     [0,0,0,0,0,1],
                     [1,1,0,0,0,0],
                     [0,1,1,0,0,0],
                     [0,0,1,1,0,0],
                     [0,0,0,1,1,0],
                     [0,0,0,0,1,1],
                     [1,0,0,0,0,1]], dtype=FLOATX)

    targets = np.array([0,
                        0,
                        1,
                        1,
                         2,
                         2,
                         2,
                         2,
                         1,
                         1,
                         0,
                         0], dtype=FLOATX)

    train_set = valid_set = test_set =[(map(lambda a: np.array([a]),x)) for x in zip(data, targets, data, data)]


    start_time = time.time()
    for k in list(trainer.train(train_set, valid_set, test_set=test_set)):
        pass
    print k
    trainer.test(0, test_set)
    end_time = time.time()


    print "elapsed time:", (end_time - start_time) / 60, "mins"