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
from basic_nn import NeuralNetwork

logging = loggers.getLogger(__name__)


class MultiRNNLayer(NeuralLayer):

    def __init__(self, size, activation='sigmoid', noise=0., dropouts=0., update_h0=False):
        """
        Simple RNN Layer, input x sequence, output y sequence, cost, update parameters.
        Train a RNN without BPTT layers, which means the history_len should be set to 0 for the training data.
        :return:
        """
        super(MultiRNNLayer, self).__init__(size, activation, noise, dropouts)
        self.learning_rate = 0.1
        self.update_h0 = update_h0

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

    def _stepping_updates(self, s, k_t):
        cost = self._cost_func(s, k_t)
        lr = self.learning_rate
        return { self.W_i: self.W_i - lr * T.grad(cost, self.W_i),
                 self.W_r: self.W_r - lr * T.grad(cost, self.W_r),
                 self.W_s: self.W_s - lr * T.grad(cost, self.W_s),
                 self.B_s: self.B_s - lr * T.grad(cost, self.B_s),
                 self.B_r: self.B_r - lr * T.grad(cost, self.B_r)}

    def _cost_func(self, s ,k_t):
        return -T.log(s[k_t])

    def _recurrent_func(self):

        def recurrent_step(x_t, k_t, h_t):
            h = self._activation_func(T.dot(x_t, self.W_i)+ T.dot(h_t, self.W_r) + self.B_r)
            s = self._softmax_func(T.dot(h, self.W_s) + self.B_s)
            return [h ,s], self._stepping_updates(s, k_t)

        [h_list, s_list], updates = theano.scan(fn=recurrent_step, sequences=[self.x, self._vars.k], outputs_info=[self.h0, None],
                                          n_steps=self.x.shape[0])

        return h_list, s_list, updates

    def _predict_func(self):

        def predict_step(x_t, h_t):
            h = self._activation_func(T.dot(x_t, self.W_i)+ T.dot(h_t, self.W_r) + self.B_r)
            s = self._softmax_func(T.dot(h, self.W_s) + self.B_s)
            return [h ,s]
        [h_list, s_list], updates = theano.scan(fn=predict_step, sequences=[self.x], outputs_info=[self.h0, None],
                                          n_steps=self.x.shape[0])
        return s_list, [(self.h0, h_list[-1])]

    def _setup_functions(self):
        self._activation_func = nnprocessors.build_activation(self.activation)
        self._softmax_func = nnprocessors.build_activation('softmax')
        self.hidden_func, self.output_func, updates = self._recurrent_func()
        self.predict_func, self.predict_updates = self._predict_func()
        self.monitors.append(("h<0.1", 100 * (abs(self.hidden_func[-1]) < 0.1).mean()))
        self.monitors.append(("h<0.9", 100 * (abs(self.hidden_func[-1]) < 0.9).mean()))
        if self.update_h0:
          self.updates.append((self.h0, self.hidden_func[-1]))
        self.updates.extend(updates.items())

    def _setup_params(self):
        self.h0 = theano.shared(value=np.zeros((self.output_n,), dtype=FLOATX), name='h_input')

        self.W_i, _, self.param_count = self.create_params(self.input_n, self.output_n, "input")
        self.W_r, self.B_r, param_count = self.create_params(self.output_n, self.output_n, "recurrent")
        self.param_count += param_count
        self.W_s, self.B_s, param_count = self.create_params(self.output_n, self.input_n, "softmax")
        self.param_count += param_count

        # Don't register parameters to the whole network
        # Update inside the recurrent steps
        self.W = []
        self.B = []


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


class MultiLayerRNN(NeuralNetwork):

    def __init__(self, config):
        super(MultiLayerRNN, self).__init__(config)
        self._predict_compiled = False

    def setup_vars(self):
        super(MultiLayerRNN, self).setup_vars()

        # for a classifier, k specifies the correct labels for a given input.
        self.vars.k = T.ivector('k')
        self.inputs.append(self.vars.k)

    @property
    def cost(self):
        return -T.sum(T.log(self.vars.y)[T.arange(self.vars.k.shape[0]), self.vars.k])

    @property
    def errors(self):
        return 100 * T.mean(T.neq(T.argmax(self.vars.y, axis=1), self.vars.k))

    @property
    def monitors(self):
        yield 'err', self.errors
        for name, exp in self.special_monitors:
            yield name, exp

    def _compile(self):
        if not self._predict_compiled:
            rnn_layer = self.layers[0]
            self._predict_rnn = theano.function([self.vars.x], [rnn_layer.predict_func])
            self._predict_compiled = True

    def classify(self, x):
        self._compile()
        return np.argmax(self._predict_rnn(x)[0], axis=1)
