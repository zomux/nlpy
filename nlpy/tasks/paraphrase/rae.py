#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 NLPY.ORG
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


import time, os, sys
from nlpy.lm import Vocab
from nlpy.lm.data_generator import RNNDataGenerator
from nlpy.util import internal_resource

import numpy as np
from nlpy.deep import NetworkConfig, TrainerConfig, NeuralClassifier, SGDTrainer, AdaDeltaTrainer, AdaGradTrainer
from nlpy.deep.functions import FLOATX, monitor_var_sum as MVS, plot_hinton, \
    make_float_vectors, replace_graph as RG, monitor_var as MV, \
    smart_replace_graph as SRG
from nlpy.deep.networks import NeuralLayer
from nlpy.deep.networks.recursive import RAELayer, GeneralAutoEncoder
from nlpy.deep.networks.classifier_runner import NeuralClassifierRunner
from nlpy.util import LineIterator, FakeGenerator
from nlpy.deep import nnprocessors
from nlpy.deep.trainers.optimize import optimize_parameters
from nlpy.deep.trainers.minibatch_optimizer import MiniBatchOptimizer
from collections import Counter
import copy
import numpy as np
import random as rnd
import cPickle as pickle
import theano
import theano.tensor as T
from theano.ifelse import ifelse

import logging
logging.basicConfig(level=logging.INFO)

random = rnd.Random(3)

"""
Paraphrase RAE Implementation
"""
class TreeRAELayer(NeuralLayer):

    def __init__(self, size, activation='tanh', noise=0., dropouts=0., beta=0.,
                 optimization="ADAGRAD", unfolding=True, additional_h=False, max_reg=4, deep=False, batch_size=10,
                 realtime_update=True):
        """
        Recursive autoencoder layer follows the path of a given parse tree.
        Manually accumulate gradients.
        """
        super(TreeRAELayer, self).__init__(size, activation, noise, dropouts)
        self.learning_rate = 0.01
        self.disable_bias = True
        self.optimization = optimization
        self.beta = beta
        self.unfolding = unfolding
        self.additional_h = additional_h
        self.max_reg = max_reg
        self.deep = deep
        self.batch_size = batch_size
        self.realtime_update = realtime_update
        self.encode_optimizer = MiniBatchOptimizer(batch_size=self.batch_size, realtime=realtime_update)
        self.decode_optimizer = MiniBatchOptimizer(batch_size=self.batch_size, realtime=realtime_update)

    def connect(self, config, vars, x, input_n, id="UNKNOWN"):
        """
        Connect to a network
        :type config: nlpy.deep.conf.NetworkConfig
        :type vars: nlpy.deep.functions.VarMap
        :return:
        """
        self._config = config
        self._vars = vars
        self.input_n = input_n
        self.id = id
        self.x = x
        self._setup_params()
        self._setup_functions()
        self.connected = True

    def updating_callback(self):
        if not self.realtime_update:
            self.encode_optimizer.run()
            self.decode_optimizer.run()

    def _recursive_func(self):
        seq_len = self._vars.seq.shape[0]
        # Encoding
        [reps, inter_reps, left_subreps, right_subreps, _, rep_gradients, distances], decoding_updates = theano.scan(self._recursive_step, sequences=[T.arange(seq_len)],
                                           outputs_info=[None, None, None, None, self.init_registers, None, None],
                                           non_sequences=[self.x, self._vars.seq, self._vars.back_routes, self._vars.back_lens])
        self.learning_updates.extend(decoding_updates.items())
        # Backpropagate through structure
        [g_we1s, g_we2s, g_bes, g_wee, g_bee, _], _ = theano.scan(self._bpts_step,
                                                    sequences=[T.arange(seq_len - 1, -1, -1),],
                                                    outputs_info=[None, None, None, None, None, self.init_registers],
                                                    non_sequences=[self._vars.seq, reps, inter_reps, left_subreps, right_subreps,
                                                                   rep_gradients])
        # Encoding updates
        optimize_params = [self.W_e1, self.W_e2, self.B_e]
        optimize_gradients = [T.sum(g_we1s, axis=0), T.sum(g_we2s, axis=0), T.sum(g_bes, axis=0)]
        if self.deep:
            optimize_params.extend([self.W_ee, self.B_ee])
            optimize_gradients.extend([T.sum(g_wee, axis=0), T.sum(g_bee, axis=0)])

        encoding_updates = self.encode_optimizer.setup(optimize_params, optimize_gradients, method=self.optimization,
                                                beta=self.beta, count=seq_len)
        self.learning_updates.extend(encoding_updates)
        return reps[-1], T.sum(distances)

    def _bpts_step(self, i, gradient_reg, seqs, reps, inter_reps, left_subreps, right_subreps, rep_gradients):
        # BPTS
        seq = seqs[i]
        left, right, target = seq[0], seq[1], seq[2]

        left_is_token = T.lt(left, 0)
        right_is_token = T.lt(right, 0)

        bpts_gradient = gradient_reg[target]
        rep_gradient = rep_gradients[i] + bpts_gradient

        if self.deep:
            # Implementation note:
            # As the gradient of deep encoding func wrt W_ee includes the input representation.
            # If we let T.grad to find that input representation directly, it will stuck in an infinite loop.
            # So we must use SRG in this case.
            _fake_input_rep, = make_float_vectors("_fake_input_rep")
            deep_rep = self._deep_encode(_fake_input_rep)

            node_map = {deep_rep: reps[i], _fake_input_rep: inter_reps[i]}

            g_wee = SRG(T.grad(T.sum(deep_rep), self.W_ee), node_map) * rep_gradient
            g_bee = SRG(T.grad(T.sum(deep_rep), self.B_ee), node_map) * rep_gradient
            g_inter_rep = SRG(T.grad(T.sum(deep_rep), _fake_input_rep), node_map) * rep_gradient
            inter_rep = inter_reps[i]

        else:
            g_wee = T.constant(0)
            g_bee = T.constant(0)
            g_inter_rep = rep_gradient
            inter_rep = reps[i]

        # Accelerate computation by using saved internal values.
        # For the limitation of SRG, known_grads can not be used here.
        _fake_left_rep, _fake_right_rep = make_float_vectors("_fake_left_rep", "_fake_right_rep")
        rep_node = self._encode_computation(_fake_left_rep, _fake_right_rep)
        if self.deep:
            rep_node = self._deep_encode(rep_node)

        node_map = {_fake_left_rep: left_subreps[i], _fake_right_rep: right_subreps[i], rep_node: inter_rep}

        g_we1 = SRG(T.grad(T.sum(rep_node), self.W_e1), node_map) * g_inter_rep
        g_we2 = SRG(T.grad(T.sum(rep_node), self.W_e2), node_map) * g_inter_rep
        g_be = SRG(T.grad(T.sum(rep_node), self.B_e), node_map) * g_inter_rep

        g_left_p = SRG(T.grad(T.sum(rep_node), _fake_left_rep), node_map) * g_inter_rep
        g_right_p = SRG(T.grad(T.sum(rep_node), _fake_right_rep), node_map) * g_inter_rep

        gradient_reg = ifelse(left_is_token, gradient_reg, T.set_subtensor(gradient_reg[left], g_left_p))
        gradient_reg = ifelse(right_is_token, gradient_reg, T.set_subtensor(gradient_reg[right], g_right_p))

        return g_we1, g_we2, g_be, g_wee, g_bee, gradient_reg

    def _encode_computation(self, left_rep, right_rep):
        return self._activation_func(T.dot(left_rep, self.W_e1) + T.dot(right_rep, self.W_e2) + self.B_e)

    def _deep_encode(self, rep):
        return self._activation_func(T.dot(rep, self.W_ee) + self.B_ee)

    def _recursive_step(self, i, regs, tokens, seqs, back_routes, back_lens):
        seq = seqs[i]
        # Encoding
        left, right, target = seq[0], seq[1], seq[2]

        left_rep = ifelse(T.lt(left, 0), tokens[-left], regs[left])
        right_rep = ifelse(T.lt(right, 0), tokens[-right], regs[right])

        rep = self._encode_computation(left_rep, right_rep)

        if self.deep:
            inter_rep = rep
            rep = self._deep_encode(inter_rep)
        else:
            inter_rep = T.constant(0)


        new_regs = T.set_subtensor(regs[target], rep)

        back_len = back_lens[i]

        back_reps, lefts, rights = self._unfold(back_routes[i], new_regs, back_len)
        gf_W_d1, gf_W_d2, gf_B_d1, gf_B_d2, distance, rep_gradient = self._unfold_gradients(back_reps, lefts, rights, back_routes[i],
                                                                    tokens, back_len)

        return ([rep, inter_rep, left_rep, right_rep, new_regs, rep_gradient, distance],
                self.decode_optimizer.setup([self.W_d1, self.W_d2, self.B_d1, self.B_d2],
                                    [gf_W_d1, gf_W_d2, gf_B_d1, gf_B_d2], method=self.optimization, beta=self.beta))


    def _unfold(self, back_route, regs, n, rep=None):
        if rep:
            regs = T.set_subtensor(regs[back_route[-1][2]], rep)
        [reps, lefts, rights, _], _ = theano.scan(self._decode_step, sequences=[back_route],
                                   outputs_info=[None, None, None, regs], n_steps=n)

        return reps, lefts, rights

    def _decode_computation(self, rep):
        left_dec = self._activation_func(T.dot(rep, self.W_d1) + self.B_d1)
        right_dec = self._activation_func(T.dot(rep, self.W_d2) + self.B_d2)
        return left_dec, right_dec

    def _decode_step(self, seq, regs):
        left, right, target = seq[0], seq[1], seq[2]

        left_is_not_token = T.ge(left, 0)
        right_is_not_token = T.ge(right, 0)

        rep = regs[target]

        left_dec, right_dec = self._decode_computation(rep)

        regs = ifelse(left_is_not_token, T.set_subtensor(regs[left], left_dec), regs)
        regs = ifelse(right_is_not_token, T.set_subtensor(regs[right], right_dec), regs)

        return  rep, left_dec, right_dec, regs


    def _unfold_gradients(self, reps, left_decs, right_decs, back_route, tokens, n):
        [regs, g_wd1s, g_wd2s, g_bd1s, g_bd2s, distances], _ = theano.scan(self._unfold_gradients_step, sequences=[T.arange(n-1, -1, -1)],
                                                outputs_info=[self.init_registers, None, None, None, None, None],
                                                non_sequences=[reps, left_decs, right_decs, back_route, tokens])
        target_reg = back_route[0][2]
        rep_gradient = regs[-1][target_reg]

        return T.mean(g_wd1s, axis=0), T.mean(g_wd2s, axis=0), T.mean(g_bd1s, axis=0), T.mean(g_bd2s, axis=0), T.sum(distances), rep_gradient

    def _unfold_gradients_func(self, rep, dec, g_dec, target_tok, tok, w, b, unfold_idx=0):
        distance = T.sum((target_tok - dec)**2)
        g_cost_dec = T.grad(distance, dec)

        tok_is_token = T.lt(tok, 0)
        g_dec_switcher = ifelse(tok_is_token, g_cost_dec, g_dec)

        output_distance = ifelse(tok_is_token, distance, T.constant(0.0, dtype=FLOATX))

        _rep, = make_float_vectors("_rep")
        _dec = self._decode_computation(_rep)[unfold_idx]
        node_map = {_rep: rep, _dec: dec}

        g_dec_rep = SRG(T.grad(T.sum(_dec), _rep), node_map) * g_dec_switcher
        g_dec_w = SRG(T.grad(T.sum(_dec), w), node_map) * g_dec_switcher
        g_dec_b = SRG(T.grad(T.sum(_dec), b), node_map) * g_dec_switcher

        return g_dec_rep, g_dec_w, g_dec_b, output_distance

    def _unfold_gradients_step(self, i,
                               regs,
                               reps, left_decs, right_decs, route, tokens):
        left, right, target = route[i][0], route[i][1], route[i][2]

        g_left_rep, g_wd1, g_bd1, left_distance = self._unfold_gradients_func(reps[i], left_decs[i], regs[left],
                                                             tokens[-left], left, self.W_d1, self.B_d1, 0)

        g_right_rep, g_wd2, g_bd2, right_distance = self._unfold_gradients_func(reps[i], right_decs[i], regs[right],
                                                             tokens[-right], right, self.W_d2, self.B_d2, 1)


        new_regs = T.set_subtensor(regs[target], (g_left_rep + g_right_rep))

        distance = left_distance + right_distance

        return new_regs, g_wd1, g_wd2, g_bd1, g_bd2, distance


    def encode_func(self):
        seq_len = self._vars.seq.shape[0]
        # Encoding
        [reps, _], _ = theano.scan(self._encode_step, sequences=[T.arange(seq_len)],
                                           outputs_info=[None, self.init_registers],
                                           non_sequences=[self.x, self._vars.seq])

        return reps


    def _encode_step(self, i, regs, tokens, seqs):
        seq = seqs[i]
        # Encoding
        left, right, target = seq[0], seq[1], seq[2]

        left_rep = ifelse(T.lt(left, 0), tokens[-left], regs[left])
        right_rep = ifelse(T.lt(right, 0), tokens[-right], regs[right])

        rep = self._encode_computation(left_rep, right_rep)

        if self.deep:
            rep = self._deep_encode(rep)

        new_regs = T.set_subtensor(regs[target], rep)

        return rep, new_regs

    def decode_func(self):
        # Not implemented
        return T.sum(self._vars.p) + T.sum(self._vars.seq)

    def _setup_functions(self):
        self._assistive_params = []
        self._activation_func = nnprocessors.build_activation(self.activation)
        self._softmax_func = nnprocessors.build_activation('softmax')
        top_rep, self.output_func = self._recursive_func()
        # self.predict_func, self.predict_updates = self._encode_func()
        self.monitors.append(("top_rep<0.1", 100 * (abs(top_rep) < 0.1).mean()))
        self.monitors.append(("top_rep<0.9", 100 * (abs(top_rep) < 0.9).mean()))
        self.monitors.append(("top_rep:mean", abs(top_rep).mean()))

    def _setup_params(self):

        weight_scale = None

        # In this implementation, all hidden layers, terminal nodes should have same vector size
        assert self.input_n == self.output_n
        self.W_e1 = self.create_weight(self.input_n, self.output_n, "enc1", scale=weight_scale)
        self.W_e2 = self.create_weight(self.input_n, self.output_n, "enc2", scale=weight_scale)
        self.B_e = self.create_bias(self.output_n, "enc")

        self.W_d1 = self.create_weight(self.output_n, self.output_n, "dec1", scale=weight_scale)
        self.W_d2 = self.create_weight(self.output_n, self.input_n, "dec2", scale=weight_scale)
        self.B_d1 = self.create_bias(self.output_n, "dec1")
        self.B_d2 = self.create_bias(self.input_n, "dec2")

        self.init_gW_d1 = theano.shared(np.zeros_like(self.W_d1.get_value()))
        self.init_gW_d2 = theano.shared(np.zeros_like(self.W_d2.get_value()))
        self.init_gB_d1 = theano.shared(np.zeros_like(self.B_d1.get_value()))
        self.init_gB_d2 = theano.shared(np.zeros_like(self.B_d2.get_value()))

        self.h0 = None
        if self.additional_h:
            self.h0 = self.create_vector(self.output_n, "h0")

        self.W = []
        self.B = []
        self.params = [self.W_e1, self.W_e2, self.B_e, self.W_d1, self.W_d2, self.B_d1, self.B_d2]

        if self.deep:
            # Set parameters for deep encoding layer
            self.W_ee = self.create_weight(self.output_n, self.output_n, "deep_enc", scale=weight_scale)
            self.B_ee = self.create_bias(self.output_n, "deep_enc")
            self.params.extend([self.W_ee, self.B_ee])

        self.init_registers = self.create_matrix(self.max_reg + 1, self.output_n, "init_regs")
        self.zero_rep = self.create_vector(self.output_n, "zero_rep")

        # Inputs for all
        self._vars.seq = T.imatrix("seq")

        # Inputs for training
        self._vars.back_routes = T.itensor3("back_routes")
        self._vars.back_lens = T.ivector("back_lens")
        self.inputs = [self._vars.seq, self._vars.back_routes, self._vars.back_lens]

        # Just for decoding
        self._vars.n = T.iscalar("n")
        self._vars.p = T.vector("p", dtype=FLOATX)
        self.encode_inputs = [self._vars.x, self._vars.seq]
        self.decode_inputs = [self._vars.p, self._vars.seq]


"""
RAE data loader
"""
class ParaphraseDataBuilder(object):

    def __init__(self, path, max_reg=4, make_valid_data=False):
        self.max_reg = max_reg
        self.data = pickle.load(open(path))
        # random.shuffle(self.data)
        # self.data = self.data[:10000]
        for d in self.data:
            d["back_lens"] = map(len, d["back_routes"])
        if make_valid_data:
            random.shuffle(self.data)
            train_size = int(len(self.data)*0.9)
            self._train_data = self.data[:train_size]
            self._valid_data = self.data[train_size:]
        else:
            self._train_data = self.data
            self._valid_data = []

    def _pad_routes(self, data):
        pad_item = (0,0,0)
        max_length = max(map(len, data))
        for i in range(len(data)):
            d = data[i]
            d.extend([pad_item] * (max_length - len(d)))
        return data

    def get_train_data(self):
        idx_seq = range(len(self._train_data))
        random.shuffle(idx_seq)
        for j in xrange(len(idx_seq)):
            i = idx_seq[j]
            progress = float(j) / len(idx_seq)
            sys.stdout.write("\r[{0}] {1}%".format('#'*(int(progress*30)), int(progress * 100)))
            sys.stdout.flush()
            if self._train_data[i]["max_reg"] > self.max_reg:
                continue
            yield (self._train_data[i]["token_data"], self._train_data[i]["sequence"],
                   self._pad_routes(self._train_data[i]["back_routes"]),
                   self._train_data[i]["back_lens"])
        sys.stdout.write("\r ")
        sys.stdout.flush()

    def get_valid_data(self):
        idx_seq = range(len(self._valid_data))
        random.shuffle(idx_seq)
        for i in idx_seq:
            if self._valid_data[i]["max_reg"] > self.max_reg:
                continue
            yield (self._valid_data[i]["token_data"], self._valid_data[i]["sequence"],
                   self._pad_routes(self._valid_data[i]["back_routes"]),
                   self._valid_data[i]["back_lens"])

    def train_data(self):
        return FakeGenerator(self, "get_train_data")

    def valid_data(self):
        return FakeGenerator(self, "get_valid_data")

def get_rae_network(model_path=""):
    net_conf = NetworkConfig(input_size=300)
    net_conf.layers = [TreeRAELayer(size=300)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = GeneralAutoEncoder(net_conf)
    if os.path.exists(model_path):
        network.load_params(model_path)
    return network

if __name__ == '__main__':

    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("premodel")
    ap.add_argument("model")
    args = ap.parse_args()

    print "[ARGS]", args

    train_data = ParaphraseDataBuilder(args.data).train_data()
    valid_data = ParaphraseDataBuilder("/home/hadoop/data/paraphrase/rae_valid_data2_samp.pkl").train_data()

    """
    Setup network
    """
    pretrain_model = args.premodel
    model_path = args.model

    net_conf = NetworkConfig(input_size=300)
    net_conf.layers = [TreeRAELayer(size=300, deep=True, beta=0.00001, optimization="FINETUNING_ADAGRAD",
                                    batch_size=10, realtime_update=False)]

    trainer_conf = TrainerConfig()
    trainer_conf.learning_rate = 0.01
    trainer_conf.weight_l2 = 0.0001
    trainer_conf.hidden_l2 = 0.0001
    trainer_conf.monitor_frequency = trainer_conf.validation_frequency = trainer_conf.test_frequency = 1

    network = GeneralAutoEncoder(net_conf)

    trainer = SGDTrainer(network, config=trainer_conf)

    """
    Run the network
    """
    start_time = time.time()

    if os.path.exists(pretrain_model):
        network.load_params(pretrain_model)
    # elif os.path.exists(model_path):
    #     network.load_params(model_path)

    c = 0
    for _ in trainer.train(train_data, valid_set=valid_data):
        c += 1
        if c > 20:
           pass
        pass

    end_time = time.time()
    network.save_params(model_path)

    print "elapsed time:", (end_time - start_time) / 60, "mins"

