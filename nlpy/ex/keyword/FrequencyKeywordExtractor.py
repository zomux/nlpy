#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import os
from nlpy.util import internal_resource

_FREQUENCY_DATA_PATH = internal_resource("en_us_with_coca_1m_bigram_words.txt")

class FrequencyKeywordExtractor:

    def __init__(self):
        self._build_freqmap()
        self._threshold = 500

    def _build_freqmap(self):
        self._freqmap = {}
        for freq, word in (s.strip().split("\t") for s in open(_FREQUENCY_DATA_PATH).xreadlines()):
            self._freqmap[word] = float(freq)

    def set_threshold(self, threshold):
        """
        :rtype threshold: int
        """
        self._threshold = threshold

    def threshold(self):
        return self._threshold

    def extract(self, words):
        """
        :type words: list of str
        :rtype: list of str
        """
        keywords = filter(lambda w: w in self._freqmap and self._freqmap[w] < self._threshold, words)
        keywords.sort(key=lambda w: self._freqmap[w])
        return keywords

