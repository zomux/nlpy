#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import time, os, sys, re
from nlpy.lm import Vocab
from nlpy.lm.data_generator import RNNDataGenerator
from nlpy.util import internal_resource
from nlpy.deep import NetworkConfig, TrainerConfig, NeuralClassifier, NeuralRegressor, SGDTrainer, AdaDeltaTrainer, AdaGradTrainer
from nlpy.deep.functions import FLOATX, monitor_var_sum as MVS, plot_hinton, \
    make_float_vectors, replace_graph as RG, monitor_var as MV, \
    smart_replace_graph as SRG
from nlpy.deep.networks import NeuralLayer
from nlpy.deep.networks.recursive import RAELayer, GeneralAutoEncoder
from nlpy.deep.networks.classifier_runner import NeuralClassifierRunner
from nlpy.util import LineIterator, FakeGenerator
from nlpy.deep import nnprocessors
from nlpy.deep.trainers.optimize import optimize_parameters
from collections import Counter
import copy
from nlpy.basic import DefaultTokenizer
import numpy as np
import random as rnd
import cPickle as pickle
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import gzip
import math

import logging
logging.basicConfig(level=logging.INFO)

random = rnd.Random(3)

"""
Data Generator
"""

class FeaturePreProcessor(object):

    def __init__(self, num_features=True):
        self._tokenizer = DefaultTokenizer()
        self.num_features = num_features

    def _get_num_feature(self, s1, s2):
        nums_1, nums_2 = set(), set()
        for t1 in self._tokenizer.tokenize(s1):
            if re.match(r"^[0-9.]+$", t1):
                nums_1.add(t1)
        for t2 in self._tokenizer.tokenize(s2):
            if re.match(r"^[0-9.]+$", t2):
                nums_2.add(t1)
        feat = [0, 0, 0]
        if nums_1 == nums_2 or (not nums_1 and not nums_2):
            feat[0] = 1
        for n1 in nums_1:
            if n1 in nums_2:
                feat[1] = 1
                break
        if nums_1 != nums_2 and (nums_1.issubset(nums_2) or nums_2.issubset(nums_1)):
            feat[2] = 1
        return feat

    def preprocess(self, data):
        sent1, sent2, label, input = data
        # Normalize
        # input = (input + (input > 10) * (10 - input)) / 10 - 0.5
        input = (input - np.mean(input)) / np.sqrt(np.var(input))
        input = input.flatten()
        if self.num_features:
            input = np.concatenate([input.flatten(), np.array(self._get_num_feature(sent1, sent2))])
        return [input], [label]

    def preprocess_nolabel(self, sent1, sent2, input):
        # Normalize
        input = (input + (input > 10) * (10 - input)) / 10 - 0.5
        input = input.flatten()
        if self.num_features:
            input = np.concatenate([input.flatten(), np.array(self._get_num_feature(sent1, sent2))])
        return [input]

class ParaphraseClassifierDataBuilder(object):

    def __init__(self, train_path, test_path, batch_size=10, num_features=True):
        self.num_features = num_features
        self.preprocessor = FeaturePreProcessor(num_features=num_features)
        self._raw_train_data = pickle.load(gzip.open(train_path))
        self._raw_test_data = pickle.load(gzip.open(test_path))
        self._train_data = map(self.preprocessor.preprocess, self._raw_train_data)
        random.shuffle(self._train_data)
        valid_size = int(len(self._train_data) * 0.08)
        self._valid_data = map(self.preprocessor.preprocess, self._raw_test_data)#self._train_data[:valid_size]
        self._train_data = self._train_data#[valid_size:]
        self._test_data = map(self.preprocessor.preprocess, self._raw_test_data)
        self.batch_size = batch_size

    def get_train_data(self):
        random.shuffle(self._train_data)
        mini_batchs = int(math.ceil(float(len(self._train_data)) / self.batch_size))
        for i in range(mini_batchs):
            inputs = []
            labels = []
            for input, label in self._train_data[i*self.batch_size: (i+1)*self.batch_size]:
                inputs.extend(input)
                labels.extend(label)
            yield inputs, labels

    def get_valid_data(self):
        random.shuffle(self._valid_data)
        for x in self._valid_data:
            yield x

    def get_test_data(self):
        random.shuffle(self._test_data)
        for x in self._test_data:
            yield x

    def train_data(self):
        return FakeGenerator(self, "get_train_data")

    def valid_data(self):
        return FakeGenerator(self, "get_valid_data")

    def test_data(self):
        return FakeGenerator(self, "get_test_data")


"""
Softmax network
"""

def get_classify_network(path=None):

    net_conf = NetworkConfig(input_size=903)
    net_conf.layers = [NeuralLayer(size=300, activation='tanh'), NeuralLayer(size=2, activation='softmax')]

    # net_conf.input_dropouts = 0.1
    # net_conf.input_noise = 0.05

    # net_conf.input_noise = 0
    network = NeuralClassifier(net_conf)
    if path:
        network.load_params(path)

    # network.special_monitors.append(("err_rate", T.mean(T.neq(network.vars.y > 0.5, network.vars.k > 0.5))))

    return network


if __name__ == '__main__':
    builder = ParaphraseClassifierDataBuilder("/home/hadoop/data/paraphrase/mspr/train3.pkl.gz",
                                              "/home/hadoop/data/paraphrase/mspr/test3.pkl.gz")
    # builder = ParaphraseClassifierDataBuilder("/home/hadoop/data/paraphrase/data/bank_classify_train2.pkl.gz",
    #                                           "/home/hadoop/data/paraphrase/data/bank_classify_test2.pkl.gz")

    model_path = "/home/hadoop/play/model_zoo/parahrase_classifier4.gz"
    net = get_classify_network()

    if os.path.exists(model_path) and False:
        net.load_params(model_path)

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    # trainer_conf.update_l1 = 0.0001
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    trainer = AdaDeltaTrainer(net, config=trainer_conf)

    start_time = time.time()

    c = 1
    for _ in trainer.train(builder.train_data(), builder.valid_data(), builder.test_data()):
        c += 1
        if c >= 80:
            trainer.test(0, builder.test_data())
            break
        pass

    end_time = time.time()

    net.save_params(model_path)

    print "elapsed time:", (end_time - start_time) / 60, "mins"



