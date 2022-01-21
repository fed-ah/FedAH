import math
import torch
from torch.nn import functional as F
import copy
import scipy.stats

import numpy as np

def compute_D(t1,t2):
    d = (t1 - t2).pow(2).sum()
    return d

def compute_D_cuda(t1,t2,cuda0):
    t1 = t1.to(cuda0)
    t2 = t2.to(cuda0)
    d = torch.sum(torch.pow(torch.sub(t1,t2),2))
    return float(d)

def wasserstein_distance(t1,t2):
    dists = [i for i in range(len(t1))]
    value = scipy.stats.wasserstein_distance(dists, dists, np.array(t1)+10, np.array(t2)+10)
    return value

def attention_layers_selfweight(sub_list,sigma,args):
    att_list = []
    for i in range(len(sub_list[0])):
        att_list_i = []
        for j in range(len(sub_list[0])):
            if j != i:
                att = (1 / sigma) * math.exp(-((1 / sigma) * sub_list[i][j]))
                att_list_i.append(att)
        max_value = np.max(att_list_i)
        min_value = np.min(att_list_i)
        att_list_i = [(ele - min_value) / (max_value - min_value) for ele in att_list_i]
        att_list_i = torch.tensor(att_list_i, dtype=torch.float)
        p = F.softmax(att_list_i, dim=0)
        p = p.tolist()
        p_add = []
        for k in range(i):
            p_add.append(args.self_weight* p[k])
        p_add.append(1-args.self_weight)
        for k in range(i,len(p)):
            p_add.append(args.self_weight* p[k])
        p_add = torch.tensor(p_add, dtype=torch.float)
        att_list.append(p_add)
    return att_list

def self_attention_transformer_selfweight(W,sigma,args):
    sub_list = np.zeros([len(W), len(W[0]),len(W[0])])
    for k in range(len(W)):
        layer_k_params = W[k]
        for i in range(len(layer_k_params)):
            wi = layer_k_params[i]

            for j in range(len(layer_k_params)):
                wj = layer_k_params[j]
                sub_value = wasserstein_distance(wi.cpu().numpy(),wj.cpu().numpy())
                sub_list[k][i][j] += sub_value

    sub_list = np.sum(sub_list, 0)
    att_list = copy.deepcopy(attention_layers_selfweight(sub_list, sigma,args))
    return att_list