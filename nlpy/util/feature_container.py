#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np
from line_iterator import LineIterator

class FeatureContainer(object):

    def __init__(self, path=None, dtype="libsvm"):
        self.N = 0
        self.data = np.zeros(0)
        self.targets = np.zeros(0)
        if path:
            self.read(path, dtype)

    def read(self, path, dtype="libsvm"):
        """
        Read feature matrix from data
        :param path: data path
        :param type: libsvm (only)
        """
        ys = []
        xs = []
        for line in LineIterator(path):
            items = line.split(" ")
            feature_map = {}
            y = 0
            for item in items:
                if ":" in item:
                    feature_idx, value = item.split(":")
                    feature_map[int(feature_idx)] = float(value)
                else:
                    y = int(item)

            max_key = max(feature_map.keys()) if feature_map else 0
            features = []
            for fidx in range(1, max_key + 1):
                if fidx in feature_map:
                    features.append(feature_map[fidx])
                else:
                    features.append(0)
            xs.append(features)
            ys.append(y)

        self.data = np.array(xs)
        self.targets = np.array(ys)