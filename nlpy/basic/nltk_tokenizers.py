#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import nltk
from abstract_tokenizer import AbstractTokenizer

class NLTKEnglishTokenizer(AbstractTokenizer):

    def tokenize(self, sent):
        """
        :type sent: str
        :rtype: list of str
        """
        return nltk.word_tokenize(sent)

    @staticmethod
    def serve(param):
        """
        For serve web requests.
        """
        output = NLTKEnglishTokenizer().tokenize(param['input'])
        return {"output": output}