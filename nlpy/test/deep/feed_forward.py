#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import unittest
from nlpy.deep.feed_forward import Network
from nlpy.deep.trainer import SGD

import theano
import theano.tensor as T
import numpy as np

class Regressor(Network):
    '''A regressor attempts to produce a target output.'''

    def setup_vars(self):
        super(Regressor, self).setup_vars()

        # the k variable holds the target output for input x.
        self.vars.k = T.matrix('k')

    @property
    def inputs(self):
        return [self.vars.x, self.vars.k]

    @property
    def cost(self):
        err = self.vars.y - self.vars.k
        return T.mean((err * err).sum(axis=1))


class Classifier(Network):
    '''A classifier attempts to match a 1-hot target output.'''

    def __init__(self, **kwargs):
        kwargs['output_activation'] = 'softmax'
        super(Classifier, self).__init__(**kwargs)

    def setup_vars(self):
        super(Classifier, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.k = T.ivector('k')

    @property
    def inputs(self):
        return [self.x, self.k]

    @property
    def cost(self):
        return -T.mean(T.log(self.y)[T.arange(self.k.shape[0]), self.k])

    @property
    def accuracy(self):
        '''Compute the percent correct classifications.'''
        return 100 * T.mean(T.eq(T.argmax(self.y, axis=1), self.k))

    @property
    def monitors(self):
        yield 'acc', self.accuracy
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    def classify(self, x):
        return self.predict(x).argmax(axis=1)

class FeedForwardTest(unittest.TestCase):

    def test(self):
        from nlpy.dataset import HeartScaleDataset
        from nlpy.deep.conf import NetworkConfig
        from nlpy.deep import NeuralLayer
        conf = NetworkConfig(input_size=13)
        conf.layers = [NeuralLayer(10), NeuralLayer(2, 'softmax')]
        ff = Regressor(conf)
        t = SGD(ff)
        train_set = [(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13]]), np.array([[1,0]]))]
        a = [HeartScaleDataset(single_target=False).train_set()]
        b = [HeartScaleDataset(single_target=False).valid_set()]
        for k in list(t.train(a, b)):
            print k
