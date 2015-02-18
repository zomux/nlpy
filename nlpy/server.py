#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

from flask import Flask, request
import json
import urllib2
import os
from argparse import ArgumentParser
from flask import make_response
app = Flask(__name__)

def import_class(name):
    mod = __import__(".".join(name.split(".")[:-1]))
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

@app.route('/')
def index():
    """
    <pre>
    NLPY Server
    ---

    To invoke methods, access "http://[host]/[class]?[parameters]"

    # Example

        http://[host]/nlpy.basic.DefaultTokenizer?input=I'll+buy.

        http://[host]/nlpy.ex.keyword.DefaultKeywordExtractor?input=I%27ll+buy+a+cup.

    # Development

    A method named "serve" must exist in the class, which accepts one parameter.

    Like this one:

    ---------------------------------------------
        class SampleTokenizer(object):

            def tokenize(self, input):
                return input.split()

            @staticmethod
            def serve(param):
                return SampleTokenizer().tokenize(param['input'])
    ---------------------------------------------

    Good luck.
    </pre>
    """
    return index.__doc__.replace("[host]", request.host)

@app.route('/<path:class_name>', methods=['GET', 'POST'])
def runner(class_name):
    try:
        mod = import_class(class_name)
    except AttributeError:
        return "Error:<br/> %s can not be found" % class_name
    except ImportError:
        return "Error:<br/> %s can not be found" % class_name

    param = {}
    for k in request.form:
        param[k] = request.form[k].encode('utf-8', "i")
    for k in request.args:
        param[k] = urllib2.unquote(request.args[k].encode('utf-8', "i"))

    result = json.dumps(mod.serve(param))
    resp = make_response(result)
    resp.headers['Access-Control-Allow-Origin'] = "*"
    return resp

def preload():
    if os.path.exists(".nlpyserverload"):
        with open(".nlpyserverload") as fpreload:
            for l in fpreload.xreadlines():
                class_name = l.strip()
                if not class_name: continue

                try:
                    mod = import_class(class_name)
                except AttributeError:
                    raise Exception("Error: %s can not be found" % class_name)
                except ImportError:
                    raise Exception("Error: %s can not be found" % class_name)

                if "load" not in dir(mod):
                    raise Exception("Error: %s do not have static method 'load'" % class_name)

                mod.load()

if __name__ == '__main__':

    ap = ArgumentParser(__name__)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--debug", type=bool, default=True)
    args = ap.parse_args()

    # Pre-load
    preload()

    # Run server instance
    app.run(port=args.port, debug=args.debug, host="0.0.0.0")


