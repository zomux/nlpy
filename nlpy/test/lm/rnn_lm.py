#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest
import time
from nlpy.lm import Vocab
from nlpy.lm.data_generator import RNNDataGenerator
from nlpy.util import internal_resource

from nlpy.deep import NetworkConfig, TrainerConfig, NeuralClassifier, SGDTrainer
from nlpy.deep.networks import RecurrentLayers, NeuralLayer
from nlpy.deep.networks.simple_rnn import SimpleRNN, SimpleRNNLayer
from nlpy.deep.networks.multilayer_rnn import RecurrentNetwork, RecurrentLayer
from nlpy.deep.networks.classifier_runner import NeuralClassifierRunner

import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

class RNNLMTest(unittest.TestCase):

    def test_train(self):
        train_path = internal_resource("lm_test/train")
        valid_path = internal_resource("lm_test/valid")
        vocab = Vocab()
        vocab.load(train_path)

        train_data = RNNDataGenerator(vocab, train_path, target_vector=False,
                                      history_len=-1, _just_test=False, fixed_length=False, progress=True)
        valid_data = RNNDataGenerator(vocab, valid_path, target_vector=False,
                                      history_len=-1, _just_test=False, fixed_length=False, progress=True)

        net_conf = NetworkConfig(input_size=vocab.size)
        net_conf.layers = [RecurrentLayer(size=50, activation='relu')]

        trainer_conf = TrainerConfig()
        trainer_conf.learning_rate = 0.1
        trainer_conf.weight_l2 = 0.0001
        trainer_conf.hidden_l2 = 0.0001
        trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

        network = RecurrentNetwork(net_conf)
        trainer = SGDTrainer(network, config=trainer_conf)


        start_time = time.time()
        for k in list(trainer.train(train_data, valid_data)):
            pass
        print k
        end_time = time.time()
        network.save_params("/tmp/lmparam.gz")


        print "elapsed time:", (end_time - start_time) / 60, "mins"

    def _test_predict(self):
        train_path = internal_resource("lm_test/train")
        test_path = internal_resource("lm_test/test")
        vocab = Vocab()
        vocab.load(train_path)

        test_data = RNNDataGenerator(vocab, test_path, target_vector=False,
                                      history_len=-1, _just_test=True, fixed_length=False)

        net_conf = NetworkConfig(input_size=vocab.size)
        net_conf.layers = [RecurrentLayer(size=50, activation='relu')]

        network = RecurrentNetwork(net_conf)
        network.load_params("/tmp/lmparam.gz")



        for d, t in test_data:

            print map(vocab.word, np.argmax(d, axis=1))
            print map(vocab.word, t)
            print map(vocab.word, network.classify(d))
            print "-"




if __name__ == '__main__':
    unittest.main()
