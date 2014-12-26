#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import os

NLPY_ROOT = os.path.dirname(os.path.dirname(__file__))

_INTERNAL_RESOURCE_ROOT = NLPY_ROOT + os.sep + 'resources'

_EXTERNAL_RESOURCE_ROOT = _INTERNAL_RESOURCE_ROOT
if 'NLPY_EX' in os.environ:
    _EXTERNAL_RESOURCE_ROOT = os.environ['NLPY_EX']
elif 'NLPY_EXTERNAL_RESOURCE' in os.environ:
    _EXTERNAL_RESOURCE_ROOT = os.environ['NLPY_EXTERNAL_RESOURCE']
elif 'HOME' in os.environ:
    _EXTERNAL_RESOURCE_ROOT = os.path.join(os.environ['HOME'], '.nlpy_external_resources')

def internal_resource(path):
    return os.path.join(_INTERNAL_RESOURCE_ROOT, path.replace('/', os.sep))

def external_resource(path):
    return os.path.join(_EXTERNAL_RESOURCE_ROOT, path.replace('/', os.sep))


# Classes
from line_iterator import LineIterator
from nbest_list import NBestList
from feature_container import FeatureContainer
from fake_generator import FakeGenerator