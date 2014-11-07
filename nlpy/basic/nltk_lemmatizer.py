#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nltk.stem.wordnet import WordNetLemmatizer

class NLTKEnglishLemmatizer(object):

    def __init__(self):
        self._lemmatizer = WordNetLemmatizer()

    def lemmatize(self, word, pos='n'):
        """
        :type word: str
        :rtype: str
        """
        word = word.decode('utf-8')
        return self._lemmatizer.lemmatize(word, pos)
