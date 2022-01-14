# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

'''
在transformer构造的基础上，每一轮输出的地方做avg，传递给下一轮
'''

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg),
# FedAvg (--alg fedavg) and FedProx (--alg prox)
# 初始化参数使用avg
import logging
import os
import time
import json

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, get_data_transformer
from models.Update import LocalUpdate
from models.test import test_img_local_all
from models.attention_rider import self_attention, self_attention_transformer,self_attention_transformer_selfweight
from log_utils.logger import loss_logger, cfs_mtrx_logger, parameter_logger, data_logger, para_record_dir,args,attention_file
# from log_utils.logger import logset
from att_utils import igfl_server_aggregate, get_para_property
import os

# args = args_parser()
save_dir = "save_" + args.alg + '_' + args.dataset + '_' + str(args.num_users) \
               + '_' + str(args.shard_per_user) + "_" + str(args.attention)
para_dir = os.path.join(para_record_dir, args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
    args.shard_per_user) + "_" + str(args.attention) + "_" + str(args.seed))
log_file = './{}/metric.log'.format(save_dir)

if not os.path.exists('./{}'.format(save_dir)):
    os.mkdir('./{}'.format(save_dir))
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("")

if not os.path.exists(para_dir):
    os.mkdir(para_dir)

logging.basicConfig(filename=log_file, level=logging.DEBUG)

np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
# np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# logset = logset(args)
# loss_logger, cfs_mtrx_logger, parameter_logger, data_logger = logset.loggers()


if __name__ == '__main__':
    # parse args
    cuda0 = torch.device('cuda:' + str(args.gpu))
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print(args)
    loss_logger.info("Start experiment with args: \n {}".format(str(args)))
    cfs_mtrx_logger.info("Start experiment with args: \n {}".format(str(args)))
    parameter_logger.info("Start experiment with args: \n {}".format(str(args)))
    data_logger.info("Start experiment with args: \n {}".format(str(args)))

    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data_transformer(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = '/home/FedRep/data/' + args.dataset + '/data/train'
            test_path = '/home/FedRep/data/' + args.dataset + '/data/test'
        else:
            train_path = '/home/FedRep/data/' + args.dataset + '/data/train'
            test_path = '/home/FedRep/data/' + args.dataset + '/data/test'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        print("----------------dataset_test-----------------")
        print(len(dataset_test))
        print("---------------------------------------------")
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    print(args.alg)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    # ['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0', 'fc.weight', 'fc.bias', 'decoder.weight', 'decoder.bias']
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [4, 3, 0, 1]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
        else:
            w_glob_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [1, 2]]
        elif 'mnist' in args.dataset:
            w_glob_keys = [net_glob.weight_keys[i] for i in [2, 3]]
        elif 'sent140' in args.dataset:
            w_glob_keys = [net_keys[i] for i in [0, 6, 7]]
        else:
            w_glob_keys = net_keys[total_num_layers - 2:]

    if args.alg == 'fedavg' or args.alg == 'prox':
        w_glob_keys = []
    if 'sent140' not in args.dataset:
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    print(total_num_layers)
    print(
        w_glob_keys)  # ['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0', 'fc.weight', 'fc.bias']
    print(net_keys)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in w_glob_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

    # generate list of local models0106 for each user
    # net_local_list = [net_glob * len(clients)]
    w_locals = {}
    w_locals_with_global_para = {}
    for user in range(args.num_users):
        w_local_dict = {}
        for key in net_glob.state_dict().keys():
            w_local_dict[key] = net_glob.state_dict()[key]
        w_locals[user] = w_local_dict
        w_locals_with_global_para[user] = copy.deepcopy(w_local_dict)

    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    accs = []
    times = []
    accs10 = 0
    accs10_glob = 0
    start = time.time()
    # before_w = copy.deepcopy(w_locals[0]['layer_input.weight'])
    # after_w = copy.deepcopy(w_locals[0]['layer_input.weight'])

    m = max(int(args.frac * args.num_users), 1)
    # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    # print(idxs_users)
    client_sample_history = dict()
    acc_list = []
    acc_list_ = []
    att_fw = open(attention_file,"w",encoding="utf-8")
    for iter in range(args.epochs + 1):
        # print("epoch:" + str(iter))
        if iter == args.epochs:
            m = args.num_users
        data_logger.info("epoch: \n {}".format(str(iter)))

        epoch_start = time.time()
        w_glob = {}
        loss_locals = []
        # m = max(int(args.frac * args.num_users), 1)
        if iter == args.epochs:
            m = args.num_users

        parameters = {}
        # for num in range(total_num_layers):
        #     parameters.append([])
        # np.random.seed(args.seed + iter * 10)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        data_logger.info("samples: \n {}".format(",".join([str(ele) for ele in idxs_users.tolist()])))
        client_sample_history[iter] = idxs_users.tolist()
        # idxs_users = [0,1]
        w_keys_epoch = w_glob_keys  # ['rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0', 'rnn.bias_hh_l0', 'fc.weight', 'fc.bias']
        times_in = []
        total_len = 0
        index_dict = {}
        if not iter:
            uk = []
        else:
            uk = np.array(list(itertools.chain.from_iterable([np.array(i.cpu()).reshape(-1) for i in [*net_glob.state_dict().values()]])))

        for ind, idx in enumerate(idxs_users):
            index_dict[str(ind)] = idx
            # print("clients:" + str(ind) + "," + str(idx))
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    # finetune
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]],
                                        idxs=dict_users_train, indd=indd)
                else:
                    # train
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]],
                                        idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in w_locals[idx].keys():
                    if k not in w_glob_keys:
                        w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
            # print("-----------------------before------------------------------------")
            # print(len(w_locals))
            # # print(w_locals)
            # print(w_local['rnn.weight_ih_l0'])
            # print("clients:" + str(ind) + "," + str(idx))
            # print("-----------------------before------------------------------------")
            # print(w_locals[idx]['layer_input.weight'])
            # before_w = copy.deepcopy(w_locals[idx]['layer_input.weight'])

            last = iter == args.epochs
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx],
                                                  w_glob_keys=w_glob_keys, lr=args.lr, last=last, uk=uk)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, w_glob_keys=w_glob_keys,
                                                  lr=args.lr, last=last, uk=uk)

            # for k in w_locals[idx].keys():
            #     # if k not in w_glob_keys:
            #     print("-----key:%s----"%str(k))
            #     print(w_locals[idx][k].shape)
            #     # print(w_locals[idx][k])
            # print("-------------------------\n")

            loss_locals.append(copy.deepcopy(loss))
            time_train_end = time.time()
            # print("each client train model cost time:%s" % str(time_train_end - start_in))
            total_len += lens[idx]

            index = 0
            if len(w_glob) == 0:
                # w_glob = copy.deepcopy(w_local)
                for k, key in enumerate(net_glob.state_dict().keys()):
                # for key in w_glob_keys:
                #     print("----w_local:%s----" % key)
                    # print(w_local[key])
                    # print(w_local[key].shape)
                    if key in w_glob_keys:
                        # print("----w_local:%s----"%key)
                        if key not in parameters.keys():
                            parameters[key] = []
                        parameters[key].append(w_local[key])
                        index += 1
                    # w_glob[key] = w_glob[key] * lens[idx]
                    w_locals[idx][key] = w_local[key]
                    w_locals_with_global_para[idx][key] = w_local[key]
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                # for key in w_glob_keys:
                    # print("-------net_glob.keys------")
                    # print(net_glob.state_dict().keys())
                    # print("-------w_glob_keys------")
                    # print(w_glob_keys)
                    # print("----w_local:%s----" % key)
                    # print(w_local[key])

                    # print("----w_local:%s----"%key)
                    # print(w_local[key].shape)
                    if key in w_glob_keys:
                        # print("----w_local:%s----"%key)
                        # w_glob[key] = w_glob[key] * lens[idx]
                        # parameters.append([])
                        if key not in parameters.keys():
                            parameters[key] = []
                        parameters[key].append(w_local[key])
                        # print("-<>-")
                        index += 1
                    w_locals[idx][key] = w_local[key]
                    w_locals_with_global_para[idx][key] = w_local[key]
            # print("-----------------------after------------------------------------")
            # print("loss:%s" % str(loss))
            # print(w_local['rnn.weight_ih_l0'])
            # print("-----------------------after------------------------------------")
            # print("loss:%s" % str(loss))
            # print(w_locals[idx]['layer_input.weight'])
            # after_w = copy.deepcopy(w_locals[idx]['layer_input.weight'])
            # print(torch.eq(before_w, after_w))
            times_in.append(time.time() - start_in)

        # if iter == 0:
        #     # print("clients:" + str(ind) + "," + str(idx))
        #     print("-----------------------before------------------------------------")
        #     print(w_locals[0]['layer_input.weight'])
        #     before_w = copy.deepcopy(w_locals[0]['layer_input.weight'])
        # if iter == 20:
        #     print("-----------------------after------------------------------------")
        #     print(w_locals[0]['layer_input.weight'])
        #     after_w = copy.deepcopy(w_locals[0]['layer_input.weight'])
        #     # print(before_w)
        #     # print(after_w)
        #     print(torch.eq(before_w, after_w.cuda("cuda:2")))
        #     print("\n\n")
        #     print(w_locals[1]['layer_input.weight'])

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)


        attention_start = time.time()
        att_fw.write("第%s个epoch:\n"%str(iter))
        for index in range(len(w_glob_keys)):#第index层attention
            att_fw.write("第%s个层:\n" % str(index))
            # key = list(net_glob.state_dict().keys())[index]
            W = []
            # print("第%s个层:\n" % str(index))
            for k in range(index+1):
                key = w_glob_keys[k]
                # print(key)
                W.append(parameters[key])
            atts = self_attention_transformer_selfweight(W,1,args)

            for i in range(len(atts)):#第i个用户的attention向量
                att_fw.write("第%s个cilent与其他客户端的attention:\n" % str(index))
                att = atts[i]
                att_fw.write("\t".join([str(ele) for ele in att.tolist()]))
                att_fw.write("\n")
                att = att.reshape(att.shape[0],1)
                # print(att.shape)
                # print(W[0].shape)
                # 0108只更新第k层
                k = index
                key = w_glob_keys[k]
                if len(W[k][0].shape) == 2:
                    att_k = torch.Tensor.expand(att, [att.shape[0], W[k][0].shape[0] * W[k][0].shape[1]])
                    # print(att.shape)
                    att_k = att_k.reshape(len(W[k]), W[k][0].shape[0], W[k][0].shape[1])

                    # print(att.shape)
                    W[k] = torch.tensor([ele.tolist() for ele in W[k]])
                    # print(W.shape)
                    w_locals[index_dict[str(i)]][key] = torch.sum(torch.mul(att_k, W[k]).transpose(0, 1),
                                                                  dim=1)  # global部分，attention后重新赋值
                    # print("-------2-------")
                    # print(torch.sum(torch.mul(att,W),dim=1).shape)
                elif len(W[k][0].shape) == 1:
                    # print(att.shape)
                    # print(W[0].shape)
                    att_k = torch.Tensor.expand(att, [att.shape[0], W[k][0].shape[0]])
                    att_k = att_k.reshape(len(W[k]), W[k][0].shape[0])
                    W[k] = torch.tensor([ele.tolist() for ele in W[k]])
                    w_locals[index_dict[str(i)]][key] = torch.sum(torch.mul(att_k, W[k]), dim=0)
                    # print("-------1-------")
                    # print(torch.sum(torch.mul(att, W), dim=0).shape)
                elif len(W[k][0].shape) == 4:
                    # print(i)
                    # print(att.shape)
                    # print(W[0].shape)
                    att_k = torch.Tensor.expand(att, [att.shape[0],
                                                      W[k][0].shape[0] * W[k][0].shape[1] * W[k][0].shape[2] *
                                                      W[k][0].shape[3]])
                    att_k = att_k.reshape(len(W[k]), W[k][0].shape[0], W[k][0].shape[1], W[k][0].shape[2],
                                          W[k][0].shape[3])
                    W[k] = torch.tensor([ele.tolist() for ele in W[k]])
                    new_key = torch.sum(torch.mul(att_k, W[k]), dim=0)
                    # print(new_key.shape)
                    # print(w_locals[index_dict[str(i)]][key].shape)
                    w_locals[index_dict[str(i)]][key] = new_key
                parameters[key][i] = copy.deepcopy(w_locals[index_dict[str(i)]][key])
                # # 更新前k层
                # for k in range(index+1):  # weight给第k层的参数
                #     key = w_glob_keys[k]
                #     if len(W[k][0].shape) == 2:
                #         att_k = torch.Tensor.expand(att,[att.shape[0],W[k][0].shape[0]*W[k][0].shape[1]])
                #         # print(att.shape)
                #         att_k = att_k.reshape(len(W[k]),W[k][0].shape[0],W[k][0].shape[1])
                #
                #         # print(att.shape)
                #         W[k] = torch.tensor([ele.tolist() for ele in W[k]])
                #         # print(W.shape)
                #         w_locals[index_dict[str(i)]][key] = torch.sum(torch.mul(att_k,W[k]).transpose(0,1),dim=1) #global部分，attention后重新赋值
                #         # print("-------2-------")
                #         # print(torch.sum(torch.mul(att,W),dim=1).shape)
                #     elif len(W[k][0].shape) == 1:
                #         # print(att.shape)
                #         # print(W[0].shape)
                #         att_k = torch.Tensor.expand(att, [att.shape[0], W[k][0].shape[0]])
                #         att_k = att_k.reshape(len(W[k]), W[k][0].shape[0])
                #         W[k] = torch.tensor([ele.tolist() for ele in W[k]])
                #         w_locals[index_dict[str(i)]][key] = torch.sum(torch.mul(att_k,W[k]),dim=0)
                #         # print("-------1-------")
                #         # print(torch.sum(torch.mul(att, W), dim=0).shape)
                #     elif len(W[k][0].shape) == 4:
                #         # print(i)
                #         # print(att.shape)
                #         # print(W[0].shape)
                #         att_k = torch.Tensor.expand(att, [att.shape[0], W[k][0].shape[0]*W[k][0].shape[1]*W[k][0].shape[2]*W[k][0].shape[3]])
                #         att_k = att_k.reshape(len(W[k]), W[k][0].shape[0],W[k][0].shape[1],W[k][0].shape[2],W[k][0].shape[3])
                #         W[k] = torch.tensor([ele.tolist() for ele in W[k]])
                #         new_key = torch.sum(torch.mul(att_k, W[k]), dim=0)
                #         # print(new_key.shape)
                #         # print(w_locals[index_dict[str(i)]][key].shape)
                #         w_locals[index_dict[str(i)]][key] = new_key
                #     parameters[key][i] = copy.deepcopy(w_locals[index_dict[str(i)]][key])

        for ind, idx in enumerate(idxs_users):
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_locals[idx])
                for k, key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = (w_glob[key] * lens[idx]).to(cuda0)
            else:
                for k, key in enumerate(net_glob.state_dict().keys()):
                    if key in w_glob_keys:
                        w_glob[key] += (w_locals[idx][key] * lens[idx]).to(cuda0)
                    else:
                        w_glob[key] += (w_locals[idx][key] * lens[idx]).to(cuda0)
        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)


        # inner_product, norm_2, cosine = get_para_property(w_locals, list(range(args.num_users)),
        #                                                   net_glob.state_dict().keys(), args,w_glob=w_glob)

        # writer_path = os.path.join(para_dir, "inner_product_round_{}.json".format(iter))
        # with open(writer_path, "w") as writer:
        #     writer.write(json.dumps(inner_product))
        #
        # writer_path = os.path.join(para_dir, "norm_2_round_{}.json".format(iter))
        # with open(writer_path, "w") as writer:
        #     writer.write(json.dumps(norm_2))
        #
        # writer_path = os.path.join(para_dir, "cosine_round_{}.json".format(iter))
        # with open(writer_path, "w") as writer:
        #     writer.write(json.dumps(cosine))

        print("attention  cost time:%s" % str(time.time() - attention_start))
        # w_local = net_glob.state_dict()
        # for k in w_glob.keys():
        #     w_local[k] = w_glob[k]
        # if args.epochs != iter:
        #     net_glob.load_state_dict(w_glob)

        for idx_tmp in w_locals_with_global_para.keys():
            for k in w_glob.keys():
                w_locals_with_global_para[idx_tmp][k] = w_glob[k]

        if iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10:
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))
            acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                     w_glob_keys=w_glob_keys, w_locals=w_locals, indd=indd,
                                                     dataset_train=dataset_train, dict_users_train=dict_users_train,
                                                     return_all=False, iter=iter)
            loss_logger.info("averaged local acc of round {}: \n{}".format(iter, json.dumps(acc_test)))
            loss_logger.info("averaged local loss of round {}: \n{}".format(iter, json.dumps(loss_test)))

            # acc_test_, loss_test_ = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
            #                                             w_glob_keys=w_glob_keys, w_locals=w_locals_with_global_para, indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
            # loss_logger.info("averaged local acc of round {}: \n{}".format(str(iter) + " with para back", json.dumps(acc_test_)))
            # loss_logger.info("averaged local loss of round {}: \n{}".format(str(iter) + " with para back", json.dumps(loss_test_)))

            accs.append(acc_test)
            # for algs which learn a single global model, these are the local accuracies (computed using the locally updated versions of the global model at the end of each round)
            if iter != args.epochs:
                print('Round {:3d}, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    iter, loss_avg, loss_test, acc_test))
            else:
                # in the final round, we sample all users, and for the algs which learn a single global model, we fine-tune the head for 10 local epochs for fair comparison with FedRep
                print('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                    loss_avg, loss_test, acc_test))

                logging.info('Final Round, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}\n'.format(
                    loss_avg, loss_test, acc_test))

            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10 += acc_test / 10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                         w_locals=None, indd=indd, dataset_train=dataset_train,
                                                         dict_users_train=dict_users_train, return_all=False)
                if iter != args.epochs:
                    print(
                        'Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            iter, loss_avg, loss_test, acc_test))
                else:
                    print(
                        'Final Round, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                            loss_avg, loss_test, acc_test))
            if iter >= args.epochs - 10 and iter != args.epochs:
                accs10_glob += acc_test / 10

        if iter % args.save_every == args.save_every - 1:
            model_save_path = './save/accs_' + args.alg + '_' + args.dataset + '_' + str(args.num_users) + '_' + str(
                args.shard_per_user) + '_iter' + str(iter) + args.function + '.pt'
            torch.save(net_glob.state_dict(), model_save_path)
        print("each epoch  cost time:%s" % str(time.time() - epoch_start))

    data_logger.info("client sample history: \n{}".format(json.dumps(client_sample_history)))
    data_logger.info("client update before performance:")
    for uid in range(args.num_users):
        uid_accs = {}
        sample_list = []
        for i in range(len(acc_list)):
            uid_accs[i] = acc_list[i][uid]
            if uid in client_sample_history[i]:
                sample_list.append(str(i))
        data_logger.info("client %s sampled history:" % str(uid))
        data_logger.info(",".join(sample_list))
        data_logger.info("client %s update before performance:" % str(uid))
        data_logger.info(json.dumps(uid_accs))
        data_logger.info("------------------------------\n")

    att_fw.close()
    print('Average accuracy final 10 rounds: {}'.format(accs10))
    logging.info('Average accuracy final 10 rounds: {}'.format(accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(accs10_glob))
    end = time.time()
    print(end - start)
    print(times)
    print(accs)
    base_dir = './save/accs_' + args.alg + '_' + args.dataset + str(args.num_users) + '_' + str(
        args.shard_per_user) + '.csv'
    user_save_path = base_dir
    accs = np.array(accs)
    accs = pd.DataFrame(accs, columns=['accs'])
    accs.to_csv(base_dir, index=False)
    logging.info("loss train")
    logging.info(",".join([str(e) for e in loss_train]) + "/n")
    logging.info("accs")
    logging.info(",".join(list(accs)) + "/n")
