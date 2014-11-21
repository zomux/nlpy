#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import itertools

import logging as loggers
import numpy as np
import numpy.random as rng
import scipy.optimize
import theano
import theano.tensor as T
from nlpy.deep.conf import TrainerConfig
from trainer import NeuralTrainer, SGDTrainer


logging = loggers.getLogger(__name__)


class SimpleRNNTrainer(SGDTrainer):
    '''Stochastic gradient descent network trainer.'''

    def __init__(self, network, config=None):
        """
        Create a SGD trainer.
        :type network:
        :type config: nlpy.deep.conf.TrainerConfig
        :return:
        """
        super(SGDTrainer, self).__init__(network, config)

        self.momentum = self.config.momentum
        self.learning_rate = self.config.learning_rate

        logging.info('compiling %s learning function', self.__class__.__name__)

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=list(network.updates) + list(self.learning_updates()), allow_input_downcast=True)

    def learning_updates(self):
        for param in self.params:
            delta = self.learning_rate * T.grad(self.J, param)
            velocity = theano.shared(
                np.zeros_like(param.get_value()), name=param.name + '_vel')
            yield velocity, self.momentum * velocity - delta
            yield param, param + velocity

    def train(self, train_set, valid_set=None, test_set=None):
        '''We train over mini-batches and evaluate periodically.'''
        iteration = 0
        while True:
            if not iteration % self.config.test_frequency and test_set:
                try:
                    self.test(iteration, test_set)
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break

            if not iteration % self.validation_frequency:
                try:
                    if not self.evaluate(iteration, valid_set):
                        logging.info('patience elapsed, bailing out')
                        break
                except KeyboardInterrupt:
                    logging.info('interrupted!')
                    break

            try:
                costs = list(zip(
                    self.cost_names,
                    np.mean([self.train_minibatch(*x) for x in train_set], axis=0)))
            except KeyboardInterrupt:
                logging.info('interrupted!')
                break
            if not iteration % self.config.monitor_frequency:
                info = ' '.join('%s=%.2f' % el for el in costs)
                logging.info('monitor (iter=%i) %s', iteration + 1, info)
            iteration += 1

            yield dict(costs)

        self.set_params(self.best_params)


    def train_minibatch(self, *x):
        costs = self.learning_func(*x)
        self.network.updating_callback(zip(self.cost_names,costs))
        return costs
