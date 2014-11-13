#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from distutils.extension import Extension
from Cython.Distutils import build_ext
import pyximport; pyximport.install()
import pca
from pca import KL
import numpy as np

# pca.hey(np.array([1,2,3,4,5]))

a = KL()
for i in range(100):
    a.add(i)
a.reduce()
print a.vect