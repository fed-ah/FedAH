import math
import torch
from torch.nn import functional as F
import time
from sklearn import preprocessing
import copy
import scipy.stats

import numpy as np

def compute_D(t1,t2):
    # print((t1 - t2).pow(2))
    # print((t1-t2).pow(2).sum())
    d = (t1 - t2).pow(2).sum()
    # print(d.device)
    return d

def compute_D_cuda(t1,t2,cuda0):
    # print((t1 - t2).pow(2))
    # print((t1-t2).pow(2).sum())
    t1 = t1.to(cuda0)
    t2 = t2.to(cuda0)
    # d = (t1 - t2).pow(2).sum()
    d = torch.sum(torch.pow(torch.sub(t1,t2),2))

    # print(d.device)
    return float(d)

def attention(wi,wj,sigma):
    # print(compute_D(wi,wj))
    time1 = time.time()
    att = (1/sigma) * math.exp(-((1/sigma) * compute_D(wi,wj)))
    time2 = time.time()
    # print("attention time cost:%s"%str(time2-time1))
    return att

def attention_layers(sub_list,sigma):
    # print(compute_D(wi,wj))
    att_list = []
    for i in range(len(sub_list[0])):
        att_list_i = []
        # print("--------------")
        # print(sub_list[i])
        for j in range(len(sub_list[0])):
            att = (1 / sigma) * math.exp(-((1 / sigma) * sub_list[i][j]))
            att_list_i.append(att)
        max_value = np.max(att_list_i)
        min_value = np.min(att_list_i)
        att_list_i = [(ele - min_value) / (max_value - min_value) for ele in att_list_i]
        # print(att_list_i)
        att_list_i = torch.tensor(att_list_i, dtype=torch.float)
        # print(att_list_i)
        p = F.softmax(att_list_i, dim=0)
        att_list.append(p)
        # print(p)
        # print(p)
    return att_list

def attention_layers_selfweight(sub_list,sigma,args):
    # print(compute_D(wi,wj))
    att_list = []
    # print(sub_list)
    # print(len(sub_list[0]))
    for i in range(len(sub_list[0])):
        att_list_i = []
        # print("--------------")
        # print(sub_list[i])
        for j in range(len(sub_list[0])):
            if j != i:
                att = (1 / sigma) * math.exp(-((1 / sigma) * sub_list[i][j]))
                att_list_i.append(att)
        max_value = np.max(att_list_i)
        min_value = np.min(att_list_i)
        att_list_i = [(ele - min_value) / (max_value - min_value) for ele in att_list_i]
        # print(att_list_i)
        att_list_i = torch.tensor(att_list_i, dtype=torch.float)
        # print(att_list_i)
        p = F.softmax(att_list_i, dim=0)
        p = p.tolist()
        # print(p)
        # print(i)
        # print(p[i])
        p_add = []
        for k in range(i):
            p_add.append(args.self_weight* p[k])
        p_add.append(1-args.self_weight)
        for k in range(i,len(p)):
            p_add.append(args.self_weight* p[k])
        # p_add = [1-args.self_weight if k==i else args.self_weight* p[i] for k in range(len(p))]
        p_add = torch.tensor(p_add, dtype=torch.float)
        att_list.append(p_add)
        # print(p)
        # print(p)
    return att_list

def attention_layers_next(sub_list,sigma):
    att_list = []
    for i in range(len(sub_list)):
        att_list_i = []
        # print("--------------")
        # print(sub_list[i])
        for j in range(len(sub_list[0])):
            att = (1 / sigma) * math.exp(-((1 / sigma) * sub_list[i][j]))
            att_list_i.append(att)
        max_value = np.max(att_list_i)
        min_value = np.min(att_list_i)
        att_list_i = [(ele - min_value) / (max_value - min_value) for ele in att_list_i]
        # print(att_list_i)
        att_list_i = torch.tensor(att_list_i, dtype=torch.float)
        p = F.softmax(att_list_i, dim=0)
        att_list.append(p)
        # print(p)
        # print(p)
    return att_list

def attention_cuda(wi,wj,sigma,cuda0):
    # print(compute_D(wi,wj))
    time1 = time.time()
    att = (1/sigma) * torch.exp(-((1/sigma) * compute_D_cuda(wi,wj,cuda0)))
    time2 = time.time()
    # print("attention time in cuda cost:%s"%str(time2-time1))
    return att

def self_attention(W,sigma,args):
    att_list = []
    att_compute_list = []
    # cuda0 = torch.device('cuda:' + str(args.gpu))
    for i in range(len(W)):
        att_list_i = []
        wi = W[i]
        # print("-----i:%s------" % str(i))
        # print("-------wi---------")
        # print(wi)
        for j in range(len(W)):
            # if i <= j :
            wj = W[j]
            # print("-----j:%s------" % str(j))
            # print("-------wj---------")
            # print(wj)
            att = attention(wi,wj,sigma)
            # print(att)
                # att = attention_cuda(wi, wj, sigma,cuda0)
            # else:
            #     att = att_compute_list[j][i]
            # # print(att)
            att_list_i.append(att)
        # print("\n")
        max_value = np.max(att_list_i)
        min_value = np.min(att_list_i)
        att_list_i =[(ele-min_value)/(max_value-min_value) for ele in att_list_i]
        # print(att_list_i)
        att_compute_list.append(att_list_i)
        att_list_i = torch.tensor(att_list_i,dtype=torch.float)
        p = F.softmax(att_list_i, dim=0)

        # print(p)
        att_list.append(p)
    # print(att_list)
    return att_list

def self_attention_acc(W,sigma,args,acc_list_sample):
    att_list = []
    att_compute_list = []
    # cuda0 = torch.device('cuda:' + str(args.gpu))
    acc_min = np.min(acc_list_sample)
    acc_max = np.max(acc_list_sample)
    acc_list_sample_ = [(ele - acc_min) / (acc_max - acc_min) for ele in acc_list_sample]
    for i in range(len(W)):
        att_list_i = []
        wi = W[i]
        # print("-----i:%s------" % str(i))
        # print("-------wi---------")
        # print(wi)
        for j in range(len(W)):
            # if i <= j :
            wj = W[j]
            # print("-----j:%s------" % str(j))
            # print("-------wj---------")
            # print(wj)
            att = attention(wi,wj,sigma)*acc_list_sample_[j]
            # print(att)
                # att = attention_cuda(wi, wj, sigma,cuda0)
            # else:
            #     att = att_compute_list[j][i]
            # # print(att)
            att_list_i.append(att)
        # print("\n")
        max_value = np.max(att_list_i)
        min_value = np.min(att_list_i)
        att_list_i =[(ele-min_value)/(max_value-min_value) for ele in att_list_i]
        # print(att_list_i)
        att_compute_list.append(att_list_i)
        att_list_i = torch.tensor(att_list_i,dtype=torch.float)
        p = F.softmax(att_list_i, dim=0)
        # print(p)

        # print(p)
        att_list.append(p)
    # print(att_list)
    return att_list

#W:layers * users * param
def self_attention_transformer(W,sigma,args):
    att_list = []
    att_compute_list = []
    # cuda0 = torch.device('cuda:' + str(args.gpu))
    # sub_list = [[[0]*len(W[0])]*len(W[0]) for i in range(len(W))]
    sub_list = np.zeros([len(W), len(W[0]),len(W[0])])
    # print(len(sub_list))
    # print(len(sub_list[0]))
    # print(len(sub_list[0][0]))
    for k in range(len(W)):
        # 第k层
        layer_k_params = W[k]
        for i in range(len(layer_k_params)):
            #第i个用户
            att_list_i = []
            wi = layer_k_params[i]

            #分别与第j个用户计算参数的差值
            for j in range(len(layer_k_params)):
                wj = layer_k_params[j]
                sub_value = compute_D(wi,wj)
                sub_list[k][i][j] += sub_value
            # print(sub_list[k][i])
        # print(sub_list[0])
        # break

    sub_list = np.sum(sub_list, 0)
    att_list = copy.deepcopy(attention_layers(sub_list, sigma))
    return att_list



def wasserstein_distance(t1,t2):
    dists = [i for i in range(len(t1))]
    value = scipy.stats.wasserstein_distance(dists, dists, np.array(t1)+10, np.array(t2)+10)
    return value

# # 0108新
# #W:layers * users * param
# def self_attention_transformer_selfweight(W,sigma,args):
#     att_list = []
#     att_compute_list = []
#     # cuda0 = torch.device('cuda:' + str(args.gpu))
#     # sub_list = [[[0]*len(W[0])]*len(W[0]) for i in range(len(W))]
#     sub_list = np.zeros([len(W), len(W[0]),len(W[0])])
#     # print("sub_list shape:")
#     # print(len(sub_list))
#     # print(len(sub_list[0]))
#     # print(len(sub_list[0][0]))
#     for k in range(len(W)):
#         # 第k层
#         layer_k_params = W[k]
#         for i in range(len(layer_k_params)):
#             #第i个用户
#             att_list_i = []
#             wi = layer_k_params[i]
#
#             #分别与第j个用户计算参数的差值
#             for j in range(len(layer_k_params)):
#                 wj = layer_k_params[j]
#                 sub_value = wasserstein_distance(wi.cpu().numpy(),wj.cpu().numpy())
#                 sub_list[k][i][j] += sub_value
#             # print(sub_list[k][i])
#         # print(sub_list[0])
#         # break
#
#     sub_list = np.sum(sub_list, 0)
#     att_list = copy.deepcopy(attention_layers_selfweight(sub_list, sigma,args))
#     return att_list

# 0108旧
#W:layers * users * param
def self_attention_transformer_selfweight(W,sigma,args):
    att_list = []
    att_compute_list = []
    # cuda0 = torch.device('cuda:' + str(args.gpu))
    # sub_list = [[[0]*len(W[0])]*len(W[0]) for i in range(len(W))]
    sub_list = np.zeros([len(W), len(W[0]),len(W[0])])
    # print("sub_list shape:")
    # print(len(sub_list))
    # print(len(sub_list[0]))
    # print(len(sub_list[0][0]))
    for k in range(len(W)):
        # 第k层
        layer_k_params = W[k]
        for i in range(len(layer_k_params)):
            #第i个用户
            att_list_i = []
            wi = layer_k_params[i]

            #分别与第j个用户计算参数的差值
            for j in range(len(layer_k_params)):
                wj = layer_k_params[j]
                sub_value = compute_D(wi,wj)
                sub_list[k][i][j] += sub_value
            # print(sub_list[k][i])
        # print(sub_list[0])
        # break

    sub_list = np.sum(sub_list, 0)
    att_list = copy.deepcopy(attention_layers_selfweight(sub_list, sigma,args))
    return att_list


#W:layers * [users*2] * param
def self_attention_transformer_nextsample(W,sigma,args,cuda0):
    sub_list = np.zeros([len(W), int(args.frac * args.num_users)+1, int(args.frac * args.num_users)+1])
    for k in range(len(W)):#层数
        layer_k_params = W[k]
        query = layer_k_params[int(args.frac * args.num_users):]
        key = layer_k_params[:int(args.frac * args.num_users)]
        for i in range(len(query)):
            wi = query[i]
            for j in range(len(key)):
                wj = key[j]
                sub_value = compute_D_cuda(wi, wj,cuda0)
                sub_list[k][i][j] += sub_value
            sub_value = compute_D(wi, wi)
            sub_list[k][i][j+1] += sub_value
    sub_list = np.sum(sub_list, 0)
    att_list = copy.deepcopy(attention_layers_next(sub_list, sigma))
    return att_list



if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3]], dtype=torch.float)
    b = torch.tensor([[1, 2, 4]], dtype=torch.float)
    print(attention(a,b,1))
    print("---------")
    c = torch.randn(2,2)
    print(c)
    self_attention(c, 1)