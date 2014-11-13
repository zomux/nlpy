#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.util import internal_resource, LineIterator

# The freq data here is required to be sorted in the reverse order of frequency.
_FREQ_DATA_PATH = internal_resource("general/en_us_with_coca_1m_bigram_words.txt")

class FreqRecaser(object):

    def __init__(self):
        """
        Initialize recase map.
        """
        self._recase_map = {}
        for line in LineIterator(_FREQ_DATA_PATH):
            _, word = line.split("\t")
            low_word = word.lower()
            if low_word not in self._recase_map:
                self._recase_map[low_word] = word

    def recase(self, word):
        """
        :param word: word
        :return: recased word
        """
        low_word = word.lower()
        if low_word not in self._recase_map:
            return word
        else:
            return self._recase_map[low_word]
