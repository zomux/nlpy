#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nltk.parse.stanford import StanfordParser as NLTKStanfordParser
from nlpy.util import external_resource

class CFGParser(object):

    def parse(self, sent):
        return None

    def extract_terminals(self, tuple_tree):
        stack = [tuple_tree]
        terminals = []
        while stack:
            tree = stack.pop(0)
            _, children = tree
            if type(children) == list:
                all_terminals = reduce(lambda x,y: x and y, ([type(child[1]) != list for child in children]))
                if all_terminals:
                    terminals.extend(children)
                else:
                    stack.extend(children)
        return terminals


class StanfordCFGParser(CFGParser):

    def __init__(self):
        self.parser = NLTKStanfordParser(path_to_jar=external_resource("stanford/stanford-parser.jar"),
                                         path_to_models_jar=external_resource("stanford/stanford-parser.jar"),
                                         model_path=external_resource("stanford/englishPCFG.ser.gz"))

    def parse(self, sent):
        tree = self.parser.parse_sents([sent])[0]
        return self._build_tuples(tree)

    def _enc(self, string):
        return string.encode("utf-8")

    def _build_tuples(self, subtree):
        is_terminal = len(subtree) == 1 and type(subtree[0]) == unicode
        if is_terminal:
            return (self._enc(subtree.label()), self._enc(subtree[0]))
        else:
            return (self._enc(subtree.label()), [self._build_tuples(t) for t in subtree])


