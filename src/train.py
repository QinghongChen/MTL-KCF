import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, f1_score
from model import MODEL
import logging


logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def train(args, data_info):
    logging.info("================== training MODEL ====================")
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    kg_data = data_info[7]

    model, optimizer_kge, optimizer_cf = _init_model(args, data_info)
    
    for step in range(args.n_epoch): 
        train_data = shuffle(train_data)
        start = 0
        while start  < kg_data.shape[0]:
            kge_loss = model('get_kge_loss', *_get_feed_data(args, kg_data[start:start + args.batch_size]['h'],
                        kg_data[start:start + args.batch_size]['r'], kg_data[start:start + args.batch_size]['t']))
            optimizer_kge.zero_grad()
            kge_loss.backward()
            optimizer_kge.step()
            start += args.batch_size

        start = 0
        while start  < train_data.shape[0]:
            cf_loss = model('get_cf_loss', *_get_feed_data(args, train_data[start:start + args.batch_size]['u'],
                        train_data[start:start + args.batch_size]['i'], train_data[start:start + args.batch_size]['label']))
            optimizer_cf.zero_grad()
            cf_loss.backward()
            optimizer_cf.step()
            start += args.batch_size
        
        train_auc, train_f1 = ctr_eval(args, model, train_data)
        eval_auc, eval_f1 = ctr_eval(args, model, eval_data)
        test_auc, test_f1 = ctr_eval(args, model, test_data)
        ctr_info = 'epoch %.2d    train auc: %.4f    train f1: %.4f    eval auc: %.4f    eval f1: %.4f   test auc: %.4f    test f1: %.4f '
        logging.info(ctr_info, step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1)


def ctr_eval(args, model, data):
    auc_list = []
    f1_list = []
    model.eval()
    start = 0
    while start < data.shape[0]:
        labels = data[start:start + args.batch_size]['label'].values
        scores = model('predict', *_get_pred_data(args, data, start, start + args.batch_size))
        scores = scores.detach().cpu().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        auc_list.append(auc)

        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        f1_list.append(f1)

        start += args.batch_size
    model.train()  
    auc = float(np.mean(auc_list))
    f1 = float(np.mean(f1_list))
    return auc, f1


def _init_model(args, data_info):
    n_user = data_info[3]
    n_item = data_info[4]
    n_relation = data_info[5]
    n_entity = data_info[6]
    model = MODEL(args, n_user, n_item, n_relation, n_entity)
    if args.use_cuda:
        model.cuda()
    optimizer_kge = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    optimizer_cf = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    return model, optimizer_kge, optimizer_cf


def _get_feed_data(args, a, b, c):
    a = torch.LongTensor(a.to_numpy())
    b = torch.LongTensor(b.to_numpy())
    c = torch.LongTensor(c.to_numpy())
    if args.use_cuda:
        a = a.cuda()
        b = b.cuda()
        c = c.cuda()
    return a, b, c


def _get_pred_data(args, data, start, end):
    # origin item
    users = torch.LongTensor(data[start:end]['u'].values)
    items = torch.LongTensor(data[start:end]['i'].values)
    if args.use_cuda:
        items = items.cuda()
        users = users.cuda()
    return users, items