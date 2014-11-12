#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.dataset import AbstractDataset
import logging
import tempfile
import sys, os
import gzip
import urllib
import cPickle

logging = logging.getLogger(__name__)

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

class MnistDataset(AbstractDataset):

    def __init__(self):
        logging.info("loading minst data")
        path = os.path.join(tempfile.gettempdir(), "mnist.pkl.gz")
        if not os.path.exists(path):
            logging.info("downloading minst data")
            urllib.urlretrieve (MNIST_URL, path)
        self._train_set, self._valid_set, self._test_set = cPickle.load(gzip.open(path, 'rb'))

    def train_set(self):
        return self._train_set

    def valid_set(self):
        return self._valid_set

    def test_set(self):
        return self._test_set