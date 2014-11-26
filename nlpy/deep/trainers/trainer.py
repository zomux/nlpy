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
import sys


logging = loggers.getLogger(__name__)


def default_mapper(f, dataset, *args, **kwargs):
    '''Apply a function to each element of a dataset.'''
    return [f(x, *args, **kwargs) for x in dataset]


def ipcluster_mapper(client):
    '''Get a mapper from an IPython.parallel cluster client.'''
    view = client.load_balanced_view()
    def mapper(f, dataset, *args, **kwargs):
        def ff(x):
            return f(x, *args, **kwargs)
        return view.map(ff, dataset).get()
    return mapper


class NeuralTrainer(object):
    '''This is a base class for all trainers.'''

    def __init__(self, network, config=None):
        """
        Basic neural network trainer.
        :type network: nlpy.deep.NeuralNetwork
        :type config: nlpy.deep.conf.TrainerConfig
        :return:
        """
        super(NeuralTrainer, self).__init__()

        self.params = network.params()
        self.config = config if config else TrainerConfig()
        self.network = network

        self.J = network.J(self.config)
        self.cost_exprs = [self.J]
        self.cost_names = ['J']
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs.append(monitor)
        logging.info("monitor list: %s" % ",".join(self.cost_names))


        logging.info('compiling evaluation function')
        self.ev_cost_exprs = []
        self.ev_cost_names = []
        for i in range(len(self.cost_names)):
            if self.cost_names[i].endswith("x"):
                continue
            self.ev_cost_exprs.append(self.cost_exprs[i])
            self.ev_cost_names.append(self.cost_names[i])
        self.evaluation_func = theano.function(
            network.inputs, self.ev_cost_exprs, updates=network.updates, allow_input_downcast=True)

        self.validation_frequency = self.config.validation_frequency
        self.min_improvement = self.config.min_improvement
        self.patience = self.config.patience

        self.shapes = [p.get_value(borrow=True).shape for p in self.params]
        self.counts = [np.prod(s) for s in self.shapes]
        self.starts = np.cumsum([0] + self.counts)[:-1]
        # self.dtype = self.params[0].get_value().dtype

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = [p.get_value().copy() for p in self.params]

    # def flat_to_arrays(self, x):
    #     x = x.astype(self.dtype)
    #     return [x[o:o+n].reshape(s) for s, o, n in
    #             zip(self.shapes, self.starts, self.counts)]
    #
    # def arrays_to_flat(self, arrays):
    #     x = np.zeros((sum(self.counts), ), self.dtype)
    #     for arr, o, n in zip(arrays, self.starts, self.counts):
    #         x[o:o+n] = arr.ravel()
    #     return x

    def set_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def test(self, iteration, test_set):
        costs = list(zip(
            self.ev_cost_names,
            np.mean([self.evaluation_func(*x) for x in test_set], axis=0)))
        info = ' '.join('%s=%.2f' % el for el in costs)
        logging.info('test    (iter=%i) %s', iteration + 1, info)

    def evaluate(self, iteration, valid_set):
        costs = list(zip(
            self.ev_cost_names,
            np.mean([self.evaluation_func(*x) for x in valid_set], axis=0)))
        marker = ''
        # this is the same as: (J_i - J_f) / J_i > min improvement
        _, J = costs[0]
        if self.best_cost - J > self.best_cost * self.min_improvement:
            self.best_cost = J
            self.best_iter = iteration
            self.best_params = [p.get_value().copy() for p in self.params]
            marker = ' *'
        info = ' '.join('%s=%.2f' % el for el in costs)
        logging.info('valid   (iter=%i) %s%s', iteration + 1, info, marker)
        return iteration - self.best_iter < self.patience

    def train(self, train_set, valid_set=None):
        raise NotImplementedError


class SGDTrainer(NeuralTrainer):
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

        network_updates = list(network.updates)
        learning_updates = list(self.learning_updates())
        update_list = network_updates + learning_updates
        logging.info("network updates: %s" % " ".join(map(str, [x[0] for x in network_updates])))
        logging.info("learning updates: %s" % " ".join(map(str, [x[0] for x in learning_updates])))

        self.learning_func = theano.function(
            network.inputs,
            self.cost_exprs,
            updates=update_list, allow_input_downcast=True)

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
        # self.network.updating_callback(zip(self.cost_names,costs))
        return costs
