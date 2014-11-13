#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from abstract_dataset import AbstractDataset

class MiniBatches(AbstractDataset):

    def __init__(self, dataset, size=50):
        self.origin = dataset
        self.size = size

    def train_set(self):

        pass

    def test_set(self):
        pass

    def valid_set(self):
        pass