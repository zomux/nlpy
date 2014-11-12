#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np


class AbstractDataset(object):

    def train_set(self):
        """
        :rtype: tuple
        """

    def valid_set(self):
        """
        :rtype: tuple
        """

    def test_set(self):
        """
        :rtype: tuple
        """