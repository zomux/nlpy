#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from nltk.parse.stanford import StanfordParser as NLTKStanfordParser
from nlpy.util import external_resource
import subprocess
import re
from threading  import Thread
import sys
import time
from nltk.tree import Tree
import gzip
import cPickle as pickle
from Queue import Queue, Empty
import tempfile

ON_POSIX = 'posix' in sys.builtin_module_names

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

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


class BatchStanfordCFGParser(CFGParser):

    def __init__(self):
        self._cache = {}
        self.cmd = 'java -mx1g -cp "%s/*" edu.stanford.nlp.parser.lexparser.LexicalizedParser ' \
                   '-sentences newline -encoding UTF-8 %s - ' % (
            external_resource("stanford"),
            external_resource("stanford/englishPCFG.ser.gz")
        )

    def cache(self, sentences):
        plain_sentences = [" ".join(s) for s in sentences]
        trees = self._build_trees(plain_sentences)
        # Create map
        self._cache = {}
        for key, val in zip(plain_sentences, trees):
            self._cache[key] = val

    def parse_command(self, sentences):
        fpath = tempfile.mktemp(".input")
        fpath2 = tempfile.mktemp(".output")
        open(fpath, "w").write("\n".join([" ".join(s) for s in sentences]))
        return self.cmd + "< " + fpath + " > " + fpath2

    def save(self, path):
        pickle.dump(self._cache, gzip.open(path, "wb"))

    def load(self, path):
        self._cache = pickle.load(gzip.open(path))

    def load_output(self, sentences, path):
        plain_sentences = [" ".join(s) for s in sentences]
        trees = self._parse_trees_output(open(path).read().decode("utf-8"))
        trees = map(self._build_tuples, trees)
        self._cache = {}
        assert len(plain_sentences) == len(trees)
        for key, val in zip(plain_sentences, trees):
            self._cache[key] = val

    def parse(self, tokens):
        plain_sentence = " ".join(tokens)

        if plain_sentence in self._cache:
            return self._cache[plain_sentence]
        else:
            return None

    def _build_trees(self, plain_sentences):
        input_data = "\n".join(plain_sentences)
        pipe = subprocess.Popen(self.cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        pipe.stdin.write(input_data)
        pipe.stdin.close()
        output = pipe.stdout.read().decode("utf-8", "ignore")
        trees = self._parse_trees_output(output)
        trees = map(self._build_tuples, trees)
        return trees

    def _parse_trees_output(self, output_):
        res = []
        cur_lines = []
        for line in output_.splitlines(False):
            if line == '':
                res.append(Tree.fromstring('\n'.join(cur_lines)))
                cur_lines = []
            else:
                cur_lines.append(line)
        return res

    def _enc(self, string):
        return string.convert("utf-8")

    def _build_tuples(self, subtree):
        is_terminal = len(subtree) == 1 and (type(subtree[0]) == unicode or type(subtree[0]) == str)
        if is_terminal:
            return (self._enc(subtree.label()), self._enc(subtree[0]))
        else:
            return (self._enc(subtree.label()), [self._build_tuples(t) for t in subtree])

class StanfordCFGParser(CFGParser):

    def __init__(self):
        self.setup_pipe()

    def setup_pipe(self):
        cmd = 'java -mx1g -cp "%s/*" edu.stanford.nlp.parser.lexparser.LexicalizedParser -sentences newline -tokenized %s -' % (
            external_resource("stanford"),
            external_resource("stanford/englishPCFG.ser.gz")
        )
        self.pipe = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q = Queue()
        self.t = Thread(target=enqueue_output, args=(self.pipe.stdout, self.q))
        self.t.daemon = True
        self.t.start()


    def get_raw_result(self, toks):
        sent = " ".join(toks)
        self.pipe.stdin.write(sent + "\n")
        result = ""
        c = 0
        while True:
            try:
                result += self.q.get(timeout=.05) # or q.get(timeout=.1)
            except Empty:
                if result.endswith("\n\n"):
                    break
                else:
                    c += 1
                    if c > 20:
                        self.setup_pipe()
                        time.sleep(1)
                        return self.get_raw_result(toks)
        return result

    def _build_tuple(self, parse):
        stack = []
        i = 0
        while i  < len(parse):
            item = parse[i]
            next_item = parse[i + 1] if i < len(parse) - 1 else None
            if item == "(":
                stack.append((next_item, []))
                i += 2
            elif item == ")":
                last = stack.pop()
                if not stack:
                    return last
                stack[-1][1].append(last)
                i += 1
            else:
                # Terminal node
                tag = stack.pop()[0]
                # if item == "-LRB-":
                #     item = "("
                # elif item == "-RRB-":
                #     item = ")"
                stack[-1][1].append((tag, item))
                i += 2
        item = stack.pop()
        while stack:
            stack[-1][1].append(item)
            item = stack.pop()
        return item

    def parse(self, toks):
        new_toks = []
        for tok in toks:
            if tok == "(":
                new_toks.append("-LRB-")
            elif tok == ")":
                new_toks.append("-RRB-")
            else:
                new_toks.append(tok)
        toks = new_toks
        result = self.get_raw_result(toks)
        if not result.strip():
            return ("ROOT", [])
        result = result.replace("\n", "")
        result = re.sub("\s+", " ", result)
        parse = filter(lambda x: x.strip(), re.split(r"([() ])", result))
        return self._build_tuple(parse)

