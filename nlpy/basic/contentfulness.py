#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.util import internal_resource, LineIterator
import math

_FREQ_DATA_PATH = internal_resource("general/en_us_with_coca_1m_bigram_words.txt")

class ContentfullnessEstimator(object):

    def __init__(self):
        self._maxfreq = 3000
        self._freqmap = {}
        for line in LineIterator(_FREQ_DATA_PATH):
            freq, word = line.split("\t")
            freq = int(freq)
            if freq > self._maxfreq:
                continue
            self._freqmap[word] = freq

    def estimate(self, tokens):
        """
        Estimate contentfullness
        :param tokens:
        :return: contentfullness score
        """
        score  = 0.
        count = 0
        for token in tokens:
            if token in self._freqmap:
                freq = self._freqmap[token]
                score += (1 - (float(freq) / self._maxfreq))**3
                count += 1
        if count:
            finalscore = score / count
            if finalscore < 0.1:
                finalscore = 0.1
            return finalscore
        else:
            return 0.1