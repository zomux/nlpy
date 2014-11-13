#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from abstract_tokenizer import AbstractTokenizer
from nltk_tokenizers import NLTKEnglishTokenizer
from nltk_lemmatizer import NLTKEnglishLemmatizer
from contentfulness import ContentfullnessEstimator
from recase import FreqRecaser

DefaultTokenizer = NLTKEnglishTokenizer
DefaultLemmatizer = NLTKEnglishLemmatizer
DefaultRecaser = FreqRecaser