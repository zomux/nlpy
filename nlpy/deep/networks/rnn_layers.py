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
from layer import NeuralLayer

logging = loggers.getLogger(__name__)


class RecurrentLayers(NeuralLayer):

    def __init__(self, size, depth, activation='sigmoid', noise=0., dropouts=0.):
        """
        Create a neural layer.
        :return:
        """
        super(RecurrentLayers, self).__init__(size, activation, noise, dropouts)
        self.depth = depth

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

    def _recurrent_func(self):

        def recurrent_step(x_t, h_t):
            h = self._activation_func(T.dot(x_t, self.W_input)+ T.dot(h_t, self.W_recurrence) + self.B_recurrence)
            s = self._softmax_func(T.dot(h, self.W_softmax) + self.B_softmax)
            return [h ,s]

        [h_list, s_list], _ = theano.scan(fn=recurrent_step, sequences=self.x, outputs_info=[self.h0, None],
                                          n_steps=self.x.shape[0])

        # y_preds = T.argmax(s_list, axis=1)
        return s_list

    def _setup_functions(self):
        self._activation_func = nnprocessors.build_activation(self.activation)
        self._softmax_func = nnprocessors.build_activation('softmax')
        self.output_func = self._recurrent_func()
        self.monitors.append(("h<0.1", 100 * (abs(self.output_func) < 0.1).mean()))
        self.monitors.append(("h<0.9", 100 * (abs(self.output_func) < 0.9).mean()))

    def _setup_params(self):
        self.h0 = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='h0')

        self.W_input, _, self.param_count = self.create_params(self.input_n, self.output_n, "input")
        self.W_recurrence, self.B_recurrence, param_count = self.create_params(self.output_n, self.output_n, "recurrent")
        self.param_count += param_count
        self.W_softmax, self.B_softmax, param_count = self.create_params(self.output_n, self.input_n, "softmax")
        self.param_count += param_count

        self.W = [self.W_input, self.W_recurrence, self.W_softmax]
        self.B = [self.B_recurrence, self.B_softmax]


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

