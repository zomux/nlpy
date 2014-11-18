#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np
import theano
import theano.tensor as T
from nlpy.deep.functions import FLOATX, global_rand
from nlpy.deep import nnprocessors
import logging as loggers

logging = loggers.getLogger(__name__)


class NeuralLayer(object):

    def __init__(self, size, activation='sigmoid', noise=0., dropouts=0., shared_bias=None):
        """
        Create a neural layer.
        :return:
        """
        self.activation = activation
        self.size = size
        self.output_n = size
        self.connected = False
        self.noise = noise
        self.dropouts = dropouts
        self.shared_bias = shared_bias

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        """
        Connect to a network
        :type config: nlpy.deep.conf.NetworkConfig
        :type vars: nlpy.deep.functions.VarMap
        :return:
        """
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def _setup_functions(self):
        if self.shared_bias:
            self._vars.update_if_not_existing(self.shared_bias, self.B)
        bias = self.B if not self.shared_bias else self._vars.get(self.shared_bias)

        self._activation_func = nnprocessors.build_activation(self.activation)
        self.preact_func = T.dot(self.x, self.W) + bias
        self.output_func = nnprocessors.add_noise(
                self._activation_func(self.preact_func),
                self.noise,
                self.dropouts)

    def _setup_params(self):
        self.W, self.B, self.param_count = self.create_params(self.input_n, self.output_n, self.id)

    def create_params(self, input_n, output_n, suffix, sparse=None):
        # arr = np.random.randn(input_n, output_n) / np.sqrt(input_n + output_n)
        ws = np.asarray(global_rand.uniform(low=-np.sqrt(6. / (input_n + output_n)),
                                  high=np.sqrt(6. / (input_n + output_n)),
                                  size=(input_n, output_n)))
        if self.activation == 'sigmoid':
            ws *= 4
        if sparse is not None:
            ws *= np.random.binomial(n=1, p=sparse, size=(input_n, output_n))
        weight = theano.shared(ws.astype(FLOATX), name='W_{}'.format(suffix))

        bs =  np.zeros(output_n)
        bias = theano.shared(bs.astype(FLOATX), name='b_{}'.format(suffix))
        logging.info('weights for layer %s: %s x %s', suffix, input_n, output_n)
        return weight, bias, (input_n + 1) * output_n