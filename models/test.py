# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/test.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
import math
from collections import Counter
import json

from log_utils.logger import loss_logger, cfs_mtrx_logger, parameter_logger, data_logger, para_record_dir


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        d = int(self.idxs[item])
        image, label = self.dataset[d]
        return image, label

class DatasetSplit_leaf(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label

# 投票法
# def client_predict(net, w_locals, idx_user, dataset, global_train, args):
#     predict_list = [[] for _ in range(len(global_train))]
#     data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
#     for ind, id in enumerate(idx_user):
#         net.load_state_dict(w_locals[id])
#         net.eval()
#         for idx, (data, target) in enumerate(data_loader):
#             data, target = data.to(args.device), target.to(args.device)
#             log_probs = net(data)
#             for i, j  in enumerate(log_probs):
#                 predict_list[i].append(j.argmax().item())
#     predict_label = [max(predict_list[i], key=predict_list[i].count) for i in range(len(predict_list))]
#     return predict_label

# # 最高概率法
# def client_predict(net, w_locals, idx_user, dataset, global_train, args):
#     predict_list_value = [[] for _ in range(len(global_train))]
#     predict_list_index = [[] for _ in range(len(global_train))]
#     data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
#     for ind, id in enumerate(idx_user):
#         net.load_state_dict(w_locals[id])
#         net.eval()
#         for idx, (data, target) in enumerate(data_loader):
#             data, target = data.to(args.device), target.to(args.device)
#             log_probs = net(data)
#             for i, j  in enumerate(log_probs):
#                 predict_list_value[i].append(j.max().item())
#                 predict_list_index[i].append(j.argmax().item())
#     predict_label = [predict_list_index[i][np.array(predict_list_value[i]).argmax()] for i in range(len(predict_list_value))]
#     return predict_label

# # 高概率法、阈值
# def client_predict(net, w_locals, idx_user, dataset, global_train, args):
#     predict_list_value = [[] for _ in range(len(global_train))]
#     predict_list_index = [[] for _ in range(len(global_train))]
#     data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
#     for ind, id in enumerate(idx_user):
#         net.load_state_dict(w_locals[id])
#         net.eval()
#         for idx, (data, target) in enumerate(data_loader):
#             data, target = data.to(args.device), target.to(args.device)
#             log_probs = net(data)
#             for i, j  in enumerate(log_probs):
#                 predict_list_value[i].append(math.exp(j.max().item()))
#                 predict_list_index[i].append(j.argmax().item())
#     # 一个用户预测的值大于0.5
#     predict_list_index = [predict_list_index[i][np.array(predict_list_value[i]).argmax()] for i in range(len(predict_list_value))]
#     predict_list_value = [np.array(predict_list_value[i]).max() for i in range(len(predict_list_value))]
#     predict_label = np.array(predict_list_index)[np.where(np.array(predict_list_value)>0.5, 1, 0) != 0]
#     global_train_epoch = np.array(global_train)[np.where(np.array(predict_list_value) > 0.5, 1, 0) != 0]
#     return predict_label, global_train_epoch




# 高概率法、阈值
def client_predict(net, w_locals, idx_user, dataset, global_train, args):
    predict_list_value = [[] for _ in range(len(global_train))]
    data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
    for ind, id in enumerate(idx_user):
        net.load_state_dict(w_locals[id])
        net.eval()
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net(data)
            for i, j in enumerate(log_probs):
                if math.exp(j.max().item()) > 0.95:
                    predict_list_value[i].append(j.argmax().item())
    predict_label = [[] for _ in range(len(global_train))]
    global_train_epoch = []
    cnt = 0
    for i in range(len(predict_list_value)):
        for ind, (label,times) in enumerate(Counter(predict_list_value[i]).items()):
            if not ind:
                if times > 1:
                    predict_label[i] = label
                    global_train_epoch.append(global_train[i])
                    if dataset.targets[global_train[i]] == label:
                        cnt += 1
                    break
    if cnt > 0:
        print('predict num:',len(global_train_epoch))
        print('predict acc:',cnt/len(global_train_epoch))
    while [] in predict_label:
        predict_label.remove([])
    return predict_label, global_train_epoch

# 置信度估计法1225  分类头用一样的
def client_predict_1225(net, w_locals, idx_user, dataset, global_train, args, global_key):
    # 求平均头
    w_local = {}
    for key in [*net.state_dict().keys()]:
        if key not in global_key:
            for id in range(len(w_locals)):
                if key not in w_local:
                    w_local[key] = w_locals[id][key]
                else:
                    w_local[key] += w_locals[id][key]
            w_local[key] /= len(w_locals)
            for id in idx_user:
                w_locals[id][key] = w_local[key]

    predict_list_value = [[] for _ in range(len(global_train))]
    data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
    user_label = {}
    data_predict_res = [[[] for _ in range(10)] for _ in range(len(global_train))]
    predict_label = []
    global_train_epoch = []
    for ind, id in enumerate(idx_user):
        net.load_state_dict(w_locals[id])
        net.eval()
        label_list = []
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net(data)
            # 保存每个用户对每个数据预测的类别
            for i, j in enumerate(log_probs):
                data_predict_res[i][j.argmax().item()].append(id)
            # 取该模型的数据label分布
            for i, j in enumerate(log_probs):
                # if math.exp(j.max().item()) > 0.95 and j.argmax().item() not in label_list:
                if j.argmax().item() not in label_list:
                    label_list.append(j.argmax().item())
                if len(label_list) == 2:
                    break
            user_label[id] = label_list
    label_user_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    for k, v in user_label.items():
        if len(v) > 0:
            if k not in label_user_dict[v[0]]:
                label_user_dict[v[0]].append(k)
            if len(v) > 1:
                if k not in label_user_dict[v[1]]:
                    label_user_dict[v[1]].append(k)
    for i in range(len(data_predict_res)):
        cnt = 0
        tmp_label = None
        for label in range(10):
            if set(data_predict_res[i][label]) == set(label_user_dict[label]) and len(data_predict_res[i][label]) > 2:
                cnt += 1
                tmp_label = label
        if cnt == 1:
            predict_label.append(tmp_label)
            global_train_epoch.append(global_train[i])
    if len(predict_label) > 0:
        print('predict_label:',predict_label)
        real_label = [dataset.targets[i] for i in global_train_epoch]
        print('label:',real_label)
        cnt = 0
        for i in range(len(predict_label)):
            if int(predict_label[i]) == int(real_label[i]):
                cnt += 1
        print('predict acc:', cnt/len(predict_label))
    return predict_label, global_train_epoch


# 置信度估计法
def client_predict_1224(net, w_locals, idx_user, dataset, global_train, args):
    predict_list_value = [[] for _ in range(len(global_train))]
    data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
    user_label = {}
    data_predict_res = [[[] for _ in range(10)] for _ in range(len(global_train))]
    predict_label = []
    global_train_epoch = []
    for ind, id in enumerate(idx_user):
        net.load_state_dict(w_locals[id])
        net.eval()
        label_list = []
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net(data)
            # 保存每个用户对每个数据预测的类别
            for i, j in enumerate(log_probs):
                data_predict_res[i][j.argmax().item()].append(id)
            # 取该模型的数据label分布
            for i, j in enumerate(log_probs):
                # if math.exp(j.max().item()) > 0.95 and j.argmax().item() not in label_list:
                if j.argmax().item() not in label_list:
                    label_list.append(j.argmax().item())
                if len(label_list) == 2:
                    break
            user_label[id] = label_list
    label_user_dict = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
    for k, v in user_label.items():
        if len(v) > 0:
            if k not in label_user_dict[v[0]]:
                label_user_dict[v[0]].append(k)
            if len(v) > 1:
                if k not in label_user_dict[v[1]]:
                    label_user_dict[v[1]].append(k)
    for i in range(len(data_predict_res)):
        cnt = 0
        tmp_label = None
        for label in range(10):
            if set(data_predict_res[i][label]) == set(label_user_dict[label]) and len(data_predict_res[i][label]) > 2:
                cnt += 1
                tmp_label = label
        if cnt == 1:
            predict_label.append(tmp_label)
            global_train_epoch.append(global_train[i])
    if len(predict_label) > 0:
        print('predict_label:',predict_label)
        print('label:',[dataset.targets[i] for i in global_train_epoch])
    return predict_label, global_train_epoch

# 预测
def client_predict_test(net, w_locals, idx_user, dataset, global_train, args):
    data_loader = DataLoader(DatasetSplit(dataset, global_train), batch_size=len(global_train), shuffle=False)
    pre_value = []

    for ind, id in enumerate(idx_user):
        # net.load_state_dict(w_locals[id])
        net.eval()
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target
            log_probs = net(data)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            for i, j in enumerate(log_probs):
                pre_value.append(math.exp(j.max().item()))
        for j in [0.5, 0.6, 0.7, 0.8, 0.9]:
            cnt_class = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0], 6: [0, 0], 7: [0, 0],
                         8: [0, 0], 9: [0, 0]}
            for i in range(len(target)):
                if pre_value[i] > j:
                    cnt_class[int(target[i])][1] += 1
                    if int(y_pred[i]) == int(target[i]):
                        cnt_class[int(target[i])][0] += 1
            for k,v in cnt_class.items():
                if v[1] != 0:
                    print('%s %d class acc %s total num %d'%(str(j), k, str(float(v[0]/v[1])), v[1]))



# def test_img_local(net_g, dataset, args,idx=None,indd=None, user_idx=-1, idxs=None):
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#
#     # put LEAF data into proper format
#     if 'femnist' in args.dataset:
#         leaf=True
#         datatest_new = []
#         usr = idx
#         for j in range(len(dataset[usr]['x'])):
#             datatest_new.append((torch.reshape(torch.tensor(dataset[idx]['x'][j]),(1,28,28)),torch.tensor(dataset[idx]['y'][j])))
#     elif 'sent140' in args.dataset:
#         leaf=True
#         datatest_new = []
#         for j in range(len(dataset[idx]['x'])):
#             datatest_new.append((dataset[idx]['x'][j],dataset[idx]['y'][j]))
#     else:
#         leaf=False
#
#     if leaf:
#         data_loader = DataLoader(DatasetSplit_leaf(datatest_new,np.ones(len(datatest_new))), batch_size=args.local_bs, shuffle=False)
#     else:
#         data_loader = DataLoader(DatasetSplit(dataset,idxs), batch_size=args.local_bs,shuffle=False)
#     if 'sent140' in args.dataset:
#         hidden_train = net_g.init_hidden(args.local_bs)
#     count = 0
#     for idx, (data, target) in enumerate(data_loader):
#         if 'sent140' in args.dataset:
#             input_data, target_data = process_x(data, indd), process_y(target, indd)
#             if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
#                 break
#
#             data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
#             net_g.zero_grad()
#
#             hidden_train = repackage_hidden(hidden_train)
#             output, hidden_train = net_g(data, hidden_train)
#
#             loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
#             _, pred_label = torch.max(output.t(), 1)
#             correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
#             count += args.local_bs
#             test_loss += loss.item()
#
#         else:
#             if args.gpu != -1:
#                 data, target = data.to(args.device), target.to(args.device)
#             log_probs = net_g(data)
#             # sum up batch loss
#             test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#             y_pred = log_probs.data.max(1, keepdim=True)[1]
#             correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#
#     if 'sent140' not in args.dataset:
#         count = len(data_loader.dataset)
#     test_loss /= count
#     accuracy = 100.00 * float(correct) / count
#     return  accuracy, test_loss

# def test_img_local_all(net, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False):
#     tot = 0
#     num_idxxs = args.num_users
#     acc_test_local = np.zeros(num_idxxs)
#     loss_test_local = np.zeros(num_idxxs)
#     for idx in range(num_idxxs):
#         net_local = copy.deepcopy(net)
#         if w_locals is not None:
#             w_local = net_local.state_dict()
#             for k in w_locals[idx].keys():
#                 w_local[k] = w_locals[idx][k]
#             net_local.load_state_dict(w_local)
#         net_local.eval()
#         if 'femnist' in args.dataset or 'sent140' in args.dataset:
#             a, b =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
#             tot += len(dataset_test[dict_users_test[idx]]['x'])
#         else:
#             a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
#             tot += len(dict_users_test[idx])
#         if 'femnist' in args.dataset or 'sent140' in args.dataset:
#             acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
#             loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
#         else:
#             acc_test_local[idx] = a*len(dict_users_test[idx])
#             loss_test_local[idx] = b*len(dict_users_test[idx])
#         del net_local
#
#     if return_all:
#         return acc_test_local, loss_test_local
#     return  sum(acc_test_local)/tot, sum(loss_test_local)/tot, acc_test_local, loss_test_local


def test_img_local(net_g, dataset, args, idx=None, indd=None, user_idx=-1, idxs=None, iter=None):
    net_g.eval()
    test_loss = 0
    correct = 0
    confusion_martix = np.zeros([args.num_classes, args.num_classes])
    # put LEAF data into proper format
    if 'femnist' in args.dataset:
        leaf = True
        datatest_new = []
        usr = idxs
        for j in range(len(dataset[usr]['x'])):
            datatest_new.append(
                (torch.reshape(torch.tensor(dataset[idxs]['x'][j]), (1, 28, 28)), torch.tensor(dataset[idxs]['y'][j])))
    elif 'sent140' in args.dataset:
        leaf = True
        datatest_new = []
        # print("--------test_img_local--------")
        # print("idx:" + str(idx))
        # print("dataset[usr]:")
        # print(dataset[idx]['x'][0])
        # print(dataset[idx]['y'][0])
        # print("------------------------------")
        for j in range(len(dataset[idx]['x'])):
            datatest_new.append((dataset[idx]['x'][j], dataset[idx]['y'][j]))
        # print("----------")
        # print(len(datatest_new))
        # print("----------")
    else:
        leaf = False

    if leaf:
        data_loader = DataLoader(DatasetSplit_leaf(datatest_new, np.ones(len(datatest_new))), batch_size=args.local_bs,
                                 shuffle=False)
    else:
        data_loader = DataLoader(DatasetSplit(dataset, idxs), batch_size=args.local_bs, shuffle=False)
    if 'sent140' in args.dataset:
        hidden_train = net_g.init_hidden(args.local_bs)
    count = 0
    # print("----------data_loader-----------")
    # print(data_loader)
    # print("--------------------------------")
    for idx, (data, target) in enumerate(data_loader):
        if 'sent140' in args.dataset:
            # print("-----------sent140----------")
            # print(idx)
            # print(data)
            # print(target)
            # print("----------------------------")
            input_data, target_data = process_x(data, indd), process_y(target, indd)
            # print("-------input_data--------")
            # print(input_data)
            # print(input_data.shape)
            # print("-------------------------")
            # if args.local_bs != 1 and input_data.shape[0] > args.local_bs:
            if args.local_bs != 1 and input_data.shape[0] != args.local_bs:
                break

            data, targets = torch.from_numpy(input_data).to(args.device), torch.from_numpy(target_data).to(args.device)
            net_g.zero_grad()

            hidden_train = repackage_hidden(hidden_train)
            output, hidden_train = net_g(data, hidden_train)

            loss = F.cross_entropy(output.t(), torch.max(targets, 1)[1])
            _, pred_label = torch.max(output.t(), 1)
            correct += (pred_label == torch.max(targets, 1)[1]).sum().item()
            # count += args.local_bs
            count += min(args.local_bs, input_data.shape[0])
            # print(args.local_bs)
            # print(count)
            test_loss += loss.item() * min(args.local_bs, input_data.shape[0])

        else:
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            # print("-------y_pred--------")
            # print(type(y_pred))
            # print(y_pred.size())
            # print(y_pred)
            # print("-------target--------")
            # print(type(target))
            # print(target.size())
            # print(target)
            for tgt, prd in zip(target.tolist(), y_pred.squeeze(1).tolist()):
                confusion_martix[tgt][prd] += 1

            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    if 'sent140' not in args.dataset:
        count = len(data_loader.dataset)
    # print("count:"+str(count))
    test_loss /= count
    accuracy = 100.00 * float(correct) / count
    return accuracy, test_loss, confusion_martix


def test_img_local_all(net, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False, iter=None):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    acc_test_local_record = np.zeros(num_idxxs)
    loss_test_local_record = np.zeros(num_idxxs)
    confusion_martix_record = {i: None for i in range(num_idxxs)}
    # print("---------dataset_test---------")
    # print(dataset_test)
    # print("------------------------------")
    for idx in range(num_idxxs):
        net_local = copy.deepcopy(net)
        if w_locals is not None:
            w_local = net_local.state_dict()
            for k in w_locals[idx].keys():
                w_local[k] = w_locals[idx][k]
            net_local.load_state_dict(w_local)
        net_local.eval()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            # print("---------user----------")
            # print(dict_users_test[idx])
            a, b, confusion_martix =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
            # print(len(dataset_test[dict_users_test[idx]]['x']))
        else:
            a, b, confusion_martix = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx], iter=iter)
            tot += len(dict_users_test[idx])
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
            acc_test_local_record[idx] = a
            loss_test_local_record[idx] = b
            confusion_martix_record[idx] = confusion_martix.tolist()
            acc_test_local[idx] = a*len(dict_users_test[idx])
            loss_test_local[idx] = b*len(dict_users_test[idx])
        del net_local

    if iter is not None:
        loss_logger.info("local acc of round {}: \n{}".format(iter,
                                                              {ii : acc for ii, acc in enumerate(acc_test_local_record.tolist())}
                                                              ))
        loss_logger.info("local acc of round {}: \n{}".format(iter, json.dumps(acc_test_local_record.tolist())))
        loss_logger.info("local loss of round {}: \n{}".format(iter, json.dumps(loss_test_local_record.tolist())))
        cfs_mtrx_logger.info("local confusion martix of round {}: \n{}".format(iter, json.dumps(confusion_martix_record)))

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local)/tot, sum(loss_test_local)/tot

def test_img_local_all_fh(net, args, dataset_test, dict_users_test,w_locals=None,w_glob_keys=None, indd=None,dataset_train=None,dict_users_train=None, return_all=False, iter=None, idx=None):
    tot = 0
    num_idxxs = 1
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    acc_test_local_record = np.zeros(num_idxxs)
    loss_test_local_record = np.zeros(num_idxxs)
    confusion_martix_record = {i: None for i in range(num_idxxs)}
    # print("---------dataset_test---------")
    # print(dataset_test)
    # print("------------------------------")
    net_local = copy.deepcopy(net)
    if w_locals is not None:
        w_local = net_local.state_dict()
        for k in w_locals[idx].keys():
            w_local[k] = w_locals[idx][k]
        net_local.load_state_dict(w_local)
    net_local.eval()
    if 'femnist' in args.dataset or 'sent140' in args.dataset:
        # print("---------user----------")
        # print(dict_users_test[idx])
        # a, b, confusion_martix =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
        a, b, confusion_martix = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx],
                                                iter=iter)
        tot += len(dataset_test[dict_users_test[idx]]['x'])
        # print(len(dataset_test[dict_users_test[idx]]['x']))
    else:
        a, b, confusion_martix = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx], iter=iter)
        tot += len(dict_users_test[idx])
    if 'femnist' in args.dataset or 'sent140' in args.dataset:
        acc_test_local_record[0] = a
        loss_test_local_record[0] = b
        confusion_martix_record[0] = confusion_martix.tolist()
        acc_test_local[0] = a*len(dataset_test[dict_users_test[idx]]['x'])
        loss_test_local[0] = b*len(dataset_test[dict_users_test[idx]]['x'])
    else:
        acc_test_local_record[0] = a
        loss_test_local_record[0] = b
        confusion_martix_record[0] = confusion_martix.tolist()
        acc_test_local[0] = a*len(dict_users_test[idx])
        loss_test_local[0] = b*len(dict_users_test[idx])
    del net_local

    if iter is not None:
        loss_logger.info("local acc of round {}: \n{}".format(iter,
                                                              {ii : acc for ii, acc in enumerate(acc_test_local_record.tolist())}
                                                              ))
        print("local acc of round {}: \n{}".format(iter,{ii : acc for ii, acc in enumerate(acc_test_local_record.tolist())}))
        loss_logger.info("local acc of round {}: \n{}".format(iter, json.dumps(acc_test_local_record.tolist())))
        loss_logger.info("local loss of round {}: \n{}".format(iter, json.dumps(loss_test_local_record.tolist())))
        cfs_mtrx_logger.info("local confusion martix of round {}: \n{}".format(iter, json.dumps(confusion_martix_record)))

    if return_all:
        return acc_test_local, loss_test_local
    return sum(acc_test_local)/tot, sum(loss_test_local)/tot, acc_test_local_record.tolist()[0]
