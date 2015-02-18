#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nlpy.util import internal_resource, LineIterator
from nltk_tokenizers import NLTKEnglishTokenizer
from collections import Counter

_FREQ_DATA_PATH = internal_resource("general/en_us_with_coca_1m_bigram_words.txt")
_MASSIVE_WORD_LIST = internal_resource("general/ms_top_100k_words.txt")

class ContentfullnessEstimator(object):

    def __init__(self, source='frequency'):
        self.source = source
        if source == 'frequency':
            self._load_frequency()
        elif source == 'ranking':
            self._load_ranking()
        else:
            self._load_source()

    def _load_source(self):
        tokenizer = NLTKEnglishTokenizer()
        counter = Counter()
        for l in LineIterator(self.source):
            counter.update(map(str.lower, tokenizer.tokenize(l)))

        self._freqmap = dict(counter.items())
        self._maxfreq = sum(self._freqmap.values()) * 2 / len(self._freqmap)

    def _load_ranking(self):
        self._rank_list = []
        for l in LineIterator(_MASSIVE_WORD_LIST):
            self._rank_list.append(l)

    def _load_frequency(self):
        self._maxfreq = 3000
        self._freqmap = {}
        for line in LineIterator(_FREQ_DATA_PATH):
            freq, word = line.split("\t")
            freq = int(freq)
            if freq > self._maxfreq:
                continue
            self._freqmap[word] = freq

    def _estimate_frequency(self, tokens):
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

    def _estimate_ranking(self, tokens):
        score  = 0.
        count = 0
        for token in tokens:
            idx = self._rank_list.index(token.lower())
            if idx >= 0:
                score += (float(idx) / len(self._rank_list))
                count += 1
        if count:
            finalscore = score*10 / count
            if finalscore < 0.1:
                finalscore = 0.1
            elif finalscore > 1.:
                finalscore = 1.
            return finalscore
        else:
            return 0.1

    def _estimate_source(self, tokens):
        score  = 0.
        count = 0
        for token in tokens:
            if token in self._freqmap:
                freq = self._freqmap[token.lower()]
                score += (1 - (float(freq) / self._maxfreq))
                count += 1
        if count:
            finalscore = score / count
            if finalscore < 0.1:
                finalscore = 0.1
            elif finalscore > 1.:
                finalscore = 1.
            return finalscore
        else:
            return 0.9

    def estimate(self, tokens):
        if self.source == 'frequency':
            return self._estimate_frequency(tokens)
        elif self.source == 'ranking':
            return self._estimate_ranking(tokens)
        else:
            return self._estimate_source(tokens)

    def estimate_word(self, word):
        return self.estimate([word])