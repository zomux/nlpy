#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

SENT_MARK = "</s>"
UNK_MARK = "<unk>"

from nlpy.util import LineIterator
import numpy as np
import logging as loggers

logging = loggers.getLogger(__name__)


class Vocab(object):

    def __init__(self):
        self.vocab_map = {}
        self.size = 0
        self.add(SENT_MARK)
        self.add(UNK_MARK)

    def add(self, word):
        if word not in self.vocab_map:
            self.vocab_map[word] = self.size
            self.size += 1

    def index(self, word):
        if word in self.vocab_map:
            return self.vocab_map[word]
        else:
            return self.vocab_map[UNK_MARK]

    def binvector(self, word):
        v = np.zeros(self.size, dtype=int)
        v[self.index(word)] = 1
        return v

    def binvector_of_index(self, index):
        v = np.zeros(self.size, dtype=int)
        v[index] = 1
        return v

    def load(self, path):
        logging.info("load data from %s" % path)
        for line in LineIterator(path):
            words = line.split(" ")
            map(self.add, words)
        logging.info("vocab size: %d" % self.size)

    @property
    def sent_index(self):
        return 0

    @property
    def sent_vector(self):
        return self.binvector(SENT_MARK)

