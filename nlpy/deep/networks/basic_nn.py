#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# Some codes in this file are refactored from theanonets

import logging as loggers

import theano.tensor as T
import theano


logging = loggers.getLogger(__name__)

from nlpy.deep.functions import VarMap
from nlpy.deep import nnprocessors


class NeuralNetwork(object):

    def __init__(self, config):
        """
        :type config: nlpy.deep.conf.NetworkConfig
        :return:
        """
        self.config = config
        self.vars = VarMap()
        self.hiddens = []
        self.weights = []
        self.biases = []
        self.updates = {}

        self.layers = config.layers

        self.setup_vars()
        self.vars.y, count = self.setup_layers()

        logging.info("total network parameters: %d", count)

    def setup_vars(self):
        self.vars.x = T.matrix('x')

    def setup_layers(self):
        last_size = self.config.input_size
        parameter_count = 0
        z = nnprocessors.add_noise(
            self.vars.x,
            self.config.input_noise,
            self.config.input_dropouts)
        for i, layer in enumerate(self.layers):
            size = layer.size
            layer.connect(self.config, self.vars, z, last_size, i + 1)
            parameter_count += layer.param_count
            self.hiddens.append(layer.output_func)
            self.weights.append(layer.W)
            self.biases.append(layer.B)
            z = layer.output_func
            last_size = size
        return self.hiddens.pop(), parameter_count

    @property
    def inputs(self):
        '''Return a list of Theano input variables for this network.'''
        return [self.vars.x]

    @property
    def monitors(self):
        '''Generate a sequence of name-value pairs for monitoring the network.
        '''
        yield 'err', self.cost
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    def _compile(self):
        if getattr(self, '_compute', None) is None:
            self._compute = theano.function(
                [self.vars.x], self.hiddens + [self.vars.y], updates=self.updates)

    def params(self):
        '''Return a list of the Theano parameters for this network.'''
        params = []
        params.extend(self.weights)
        if not self.config.no_learn_biases:
            params.extend(self.biases)
        return params

    def get_weights(self, layer, borrow=False):
        return self.weights[layer].get_value(borrow=borrow)

    def get_biases(self, layer, borrow=False):
        return self.biases[layer].get_value(borrow=borrow)

    def feed_forward(self, x):
        self._compile()
        return self._compute(x)

    def predict(self, x):
        return self.feed_forward(x)[-1]

    __call__ = predict


    def J(self, train_conf):
        cost = self.cost
        if train_conf.weight_l1 > 0:
            cost += train_conf.weight_l1 * sum(abs(w).sum() for w in self.weights)
        if train_conf.weight_l2 > 0:
            cost += train_conf.weight_l2 * sum((w * w).sum() for w in self.weights)
        if train_conf.hidden_l1 > 0:
            cost += train_conf.hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)
        if train_conf.hidden_l2 > 0:
            cost += train_conf.hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.hiddens)
        if train_conf.contractive_l2 > 0:
            cost += train_conf.contractive_l2 * sum(
                T.sqr(T.grad(h.mean(axis=0).sum(), self.vars.x)).sum() for h in self.hiddens)
        return cost