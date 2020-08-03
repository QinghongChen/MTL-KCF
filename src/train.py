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
    # train_data, eval_data, test_data, n_users_entities, n_relations, triplet_sets
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    kg_data = data_info[7]

    model, optimizer_kg, optimizer, loss_func = _init_model(args, data_info)
    
    for step in range(args.n_epoch): 
        train_data = shuffle(train_data)
        start = 0
        while start  < kg_data.shape[0]:
            loss_t = model('calc_t_loss', *_get_three_data(args, kg_data[start:start + args.batch_size]['h'],
                        kg_data[start:start + args.batch_size]['r'], kg_data[start:start + args.batch_size]['t']))
            print('loss_t: {}'.format(loss_t))
            optimizer_kg.zero_grad()
            loss_t.backward()
            optimizer_kg.step()
            start += args.batch_size

        start = 0
        while start  < train_data.shape[0]:
            labels = _get_label(args, train_data[start:start + args.batch_size]['label'].values)
            scores = model('calc_score', *_get_two_data(args, train_data[start:start + args.batch_size]['u'],
                        train_data[start:start + args.batch_size]['i']))
            loss_u = loss_func(scores, labels)
            print('loss_u: {}'.format(loss_u))
        
            loss_it = model('calc_it_loss', *_get_two_data(args, train_data[start:start + args.batch_size]['u'],
                        train_data[start:start + args.batch_size]['i']))
            print('loss_it: {}'.format(loss_it))
        
            loss = loss_u + loss_it
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        scores = model('calc_score', *_get_pred_data(args, data, start, start + args.batch_size))
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
    n_users = data_info[3]
    n_items = data_info[4]
    n_relations = data_info[5]
    n_entities = data_info[6]
    model = MODEL(args, n_users, n_items, n_relations, n_entities)
    if args.use_cuda:
        model.cuda()
    optimizer_kg = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.lr,
        weight_decay = args.l2_weight,
    )
    loss_func = nn.BCELoss()
    return model, optimizer_kg, optimizer, loss_func


def _get_label(args, labels):
    labels = torch.FloatTensor(labels)
    if args.use_cuda:
        labels = labels.cuda()
    return labels

def _get_three_data(args, a, b, c):
    a = torch.LongTensor(a.to_numpy())
    b = torch.LongTensor(b.to_numpy())
    c = torch.LongTensor(c.to_numpy())
    if args.use_cuda:
        a = a.cuda()
        b = b.cuda()
        c = c.cuda()
    return a, b, c


def _get_two_data(args, a, b):
    a = torch.LongTensor(a.to_numpy())
    b = torch.LongTensor(b.to_numpy())
    if args.use_cuda:
        a = a.cuda()
        b = b.cuda()
    return a, b


def _get_one_data(args, a):
    a = torch.LongTensor(a.to_numpy())
    if args.use_cuda:
        a = a.cuda()
    return a


def _get_pred_data(args, data, start, end):
    # origin item
    users = torch.LongTensor(data[start:end]['u'].values)
    items = torch.LongTensor(data[start:end]['i'].values)
    if args.use_cuda:
        items = items.cuda()
        users = users.cuda()
    return users, items