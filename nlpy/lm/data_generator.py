#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.util import LineIterator
import numpy as np

class RNNDataGenerator(object):


    def __init__(self, vocab, data_path, target_vector=False, _just_test=False):
        """
        Generate data for training with RNN
        :type vocab: nlpy.lm.Vocab
        :type data_path: str
        :type history_len: int
        :type binvector: bool
        """
        self._vocab = vocab
        self._target_vector = target_vector
        self._just_test = _just_test

        self.trunks = []

        # Treat each sentence as a trunk
        for line in LineIterator(data_path):
            sequence = []
            for w in line.split(" "):
                sequence.append(vocab.index(w))
            sequence.append(vocab.sent_index)
            self.trunks.append(sequence)


    def __iter__(self):

        for i in xrange(len(self.trunks)):
            trunk = self.trunks[i]
            if len(trunk) <= 1:
                continue
            xs, ys = [], []
            for j in range(len(trunk) - 1):
                xs.append(self._vocab.binvector_of_index(trunk[j]))
                if self._target_vector:
                    ys.append(self._vocab.binvector_of_index(trunk[j + 1]))
                else:
                    ys.append(trunk[j + 1])

            data = [np.array(xs), np.array(ys)]
            yield(data)


            if self._just_test and i > 100:
                break







