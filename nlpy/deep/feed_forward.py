#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import theano.tensor as T
import logging as loggers
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import functools
import gzip
import numpy as np
import cPickle as pickle



logging = loggers.getLogger(__name__)

from functions import softmax, FLOATX


class Network(object):

    def __init__(self, config):
        """
        :type config: nlpy.deep.conf.NetworkConfig
        :return:
        """
        self.config = config
        self.preacts = []
        self.hiddens = []
        self.weights = []
        self.biases = []
        self.updates = {}

        self.rng = RandomStreams()

        self.layers = tuple(config.layers)
        self.tied_weights = bool(config.tied_weights)
        self.decode_from = int(config.decode_from)

        self.hidden_activation = config.hidden_activation
        self._hidden_func = self._build_activation(self.hidden_activation)
        if hasattr(self._hidden_func, '__theanets_name__'):
            logging.info('hidden activation: %s', self._hidden_func.__theanets_name__)

        self.output_activation = config.output_activation
        self._output_func = self._build_activation(self.output_activation)
        if hasattr(self._output_func, '__theanets_name__'):
            logging.info('output activation: %s', self._output_func.__theanets_name__)

        self.setup_vars()
        _, encode_count = self.setup_encoder()
        self.y, decode_count = self.setup_decoder()

        logging.info('%d total network parameters', encode_count + decode_count)

    def setup_vars(self):
        '''Setup Theano variables for our network.'''
        # x is a proxy for our network's input, and y for its output.
        self.x = T.matrix('x')

    def setup_encoder(self):
        sizes = self.check_layer_sizes()
        parameter_count = 0
        x = z = self._add_noise(
            self.x,
            self.config.input_noise,
            self.config.input_dropouts)
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            W, b, count = self.create_layer(a, b, i)
            parameter_count += count
            self.preacts.append(T.dot(z, W) + b)
            self.hiddens.append(self._add_noise(
                self._hidden_func(self.preacts[-1]),
                self.config.hidden_noise,
                self.config.hidden_dropouts))
            self.weights.append(W)
            self.biases.append(b)
            z = self.hiddens[-1]
        return x, parameter_count

    def setup_decoder(self):
        parameter_count = 0

        if self.tied_weights:
            for i in range(len(self.weights) - 1, -1, -1):
                h = self.hiddens[-1]
                a, b = self.weights[i].get_value(borrow=True).shape
                logging.info('tied weights from layer %d: %s x %s', i, b, a)
                o = theano.shared(np.zeros((a, ), FLOATX), name='b_out{}'.format(i))
                self.preacts.append(T.dot(h, self.weights[i].T) + o)
                func = self._output_func if i == 0 else self._hidden_func
                self.hiddens.append(func(self.preacts[-1]))

        else:
            B = len(self.biases) - 1
            n = self.layers[-1]
            decoders = []
            for i in range(B, B - self.decode_from, -1):
                b = self.biases[i].get_value(borrow=True).shape[0]
                Di, _, count = self.create_layer(b, n, 'out_%d' % i)
                parameter_count += count - n
                decoders.append(T.dot(self.hiddens[i], Di))
                self.weights.append(Di)
            parameter_count += n
            bias = theano.shared(np.zeros((n, ), FLOATX), name='bias_out')
            self.biases.append(bias)
            self.preacts.append(sum(decoders) + bias)
            self.hiddens.append(self._output_func(self.preacts[-1]))

        return self.hiddens.pop(), parameter_count

    def check_layer_sizes(self):
        # ensure that --layers is compatible with --tied-weights.
        sizes = self.layers[:-1]
        if self.tied_weights:
            error = 'with tied-weights, layers must be an odd-length palindrome'
            assert len(self.layers) % 2 == 1, error
            k = len(self.layers) // 2
            encode = np.asarray(self.layers[:k])
            decode = np.asarray(self.layers[k+1:])
            assert (encode == decode[::-1]).all(), error
            sizes = self.layers[:k+1]
        return sizes

    @property
    def inputs(self):
        '''Return a list of Theano input variables for this network.'''
        return [self.x]

    @property
    def monitors(self):
        '''Generate a sequence of name-value pairs for monitoring the network.
        '''
        yield 'err', self.cost
        for i, h in enumerate(self.hiddens):
            yield 'h{}<0.1'.format(i+1), 100 * (abs(h) < 0.1).mean()
            yield 'h{}<0.9'.format(i+1), 100 * (abs(h) < 0.9).mean()

    @staticmethod
    def create_layer(a, b, suffix, sparse=None):
        '''Create a layer of weights and bias values.
        Parameters
        ----------
        a : int
            Number of rows of the weight matrix -- equivalently, the number of
            "input" units that the weight matrix connects.
        b : int
            Number of columns of the weight matrix -- equivalently, the number
            of "output" units that the weight matrix connects.
        suffix : str
            A string suffix to use in the Theano name for the created variables.
            This string will be appended to 'W_' (for the weights) and 'b_' (for
            the biases) parameters that are created and returned.
        sparse : float in (0, 1)
            If given, ensure that the weight matrix for the layer has only this
            proportion of nonzero entries.
        Returns
        -------
        weight : Theano shared array
            A shared array containing Theano values representing the weights
            connecting each "input" unit to each "output" unit.
        bias : Theano shared array
            A shared array containing Theano values representing the bias
            values on each of the "output" units.
        count : int
            The number of parameters that are included in the returned
            variables.
        '''
        arr = np.random.randn(a, b) / np.sqrt(a + b)
        if sparse is not None:
            arr *= np.random.binomial(n=1, p=sparse, size=(a, b))
        weight = theano.shared(arr.astype(FLOATX), name='W_{}'.format(suffix))
        arr = 1e-3 * np.random.randn(b)
        bias = theano.shared(arr.astype(FLOATX), name='b_{}'.format(suffix))
        logging.info('weights for layer %s: %s x %s', suffix, a, b)
        return weight, bias, (a + 1) * b

    def _add_noise(self, x, sigma, rho):
        if sigma > 0 and rho > 0:
            noise = self.rng.normal(size=x.shape, std=sigma, dtype=FLOATX)
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOATX)
            return mask * (x + noise)
        if sigma > 0:
            return x + self.rng.normal(size=x.shape, std=sigma, dtype=FLOATX)
        if rho > 0:
            mask = self.rng.binomial(size=x.shape, n=1, p=1-rho, dtype=FLOATX)
            return mask * x
        return x

    def _compile(self):
        if getattr(self, '_compute', None) is None:
            self._compute = theano.function(
                [self.x], self.hiddens + [self.y], updates=self.updates)

    def _build_activation(self, act=None):
        def compose(a, b):
            c = lambda z: b(a(z))
            c.__theanets_name__ = '%s(%s)' % (b.__theanets_name__, a.__theanets_name__)
            return c
        if '+' in act:
            return functools.reduce(
                compose, (self._build_activation(a) for a in act.split('+')))
        options = {
            'tanh': T.tanh,
            'linear': lambda z: z,
            'logistic': T.nnet.sigmoid,
            'sigmoid': T.nnet.sigmoid,
            'softplus': T.nnet.softplus,
            'softmax': softmax,

            # shorthands
            'relu': lambda z: z * (z > 0),
            'trel': lambda z: z * (z > 0) * (z < 1),
            'trec': lambda z: z * (z > 1),
            'tlin': lambda z: z * (abs(z) > 1),

            # modifiers
            'rect:max': lambda z: T.minimum(1, z),
            'rect:min': lambda z: T.maximum(0, z),

            # normalization
            'norm:dc': lambda z: (z.T - z.mean(axis=1)).T,
            'norm:max': lambda z: (z.T / T.maximum(1e-10, abs(z).max(axis=1))).T,
            'norm:std': lambda z: (z.T / T.maximum(1e-10, T.std(z, axis=1))).T,
            }
        for k, v in options.items():
            v.__theanets_name__ = k
        try:
            return options[act]
        except KeyError:
            raise KeyError('unknown activation %r' % act)

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




    def J(self, weight_l1=0, weight_l2=0, hidden_l1=0, hidden_l2=0, contractive_l2=0, **unused):
        '''Return a variable representing the cost or loss for this network.
        Parameters
        ----------
        weight_l1 : float, optional
            Regularize the L1 norm of unit connection weights by this constant.
        weight_l2 : float, optional
            Regularize the L2 norm of unit connection weights by this constant.
        hidden_l1 : float, optional
            Regularize the L1 norm of hidden unit activations by this constant.
        hidden_l2 : float, optional
            Regularize the L2 norm of hidden unit activations by this constant.
        contractive_l2 : float, optional
            Regularize model using the Frobenius norm of the hidden Jacobian.
        Returns
        -------
        Theano variable
            A variable representing the overall cost value of this network.
        '''
        cost = self.cost
        if weight_l1 > 0:
            cost += weight_l1 * sum(abs(w).sum() for w in self.weights)
        if weight_l2 > 0:
            cost += weight_l2 * sum((w * w).sum() for w in self.weights)
        if hidden_l1 > 0:
            cost += hidden_l1 * sum(abs(h).mean(axis=0).sum() for h in self.hiddens)
        if hidden_l2 > 0:
            cost += hidden_l2 * sum((h * h).mean(axis=0).sum() for h in self.hiddens)
        if contractive_l2 > 0:
            cost += contractive_l2 * sum(
                T.sqr(T.grad(h.mean(axis=0).sum(), self.x)).sum() for h in self.hiddens)
        return cost
