import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MODEL(nn.Module):
    def __init__(self, args, n_user, n_item, n_relation, n_entity):
        super(MODEL, self).__init__()
        self._parse_args(args, n_user, n_item, n_relation, n_entity)
        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.item_emb = nn.Embedding(self.n_item, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)

        self.user_mlp = self.linear_layer(self.dim, self.dim, dropout=0.5)
        self.relation_mlp = self.linear_layer(self.dim, self.dim, dropout=0.5)
        self.cf_mlp = self._build_higher_mlp(self.dim * 2, 1, dropout=0.5)
        self.kge_mlp = self._build_higher_mlp(self.dim * 2, self.dim, dropout=0.5)
        
        self.weight_vv = torch.rand((self.dim, 1), requires_grad=True)
        self.weight_ev = torch.rand((self.dim, 1), requires_grad=True)
        self.weight_ve = torch.rand((self.dim, 1), requires_grad=True)
        self.weight_ee = torch.rand((self.dim, 1), requires_grad=True)
        self.bias_v = torch.rand(1, requires_grad=True)
        self.bias_e = torch.rand(1, requires_grad=True)

        self._init_weight()

    def _build_kge_loss(self, h, r, t):
        h_emb = self.entity_emb(h)
        item_emb = self.item_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        for i in range(self.L):
            r_emb = self.relation_mlp(r_emb)
            item_emb, h_emb = self.cc_unit(item_emb, h_emb)
        t_pred_emb = self.kge_mlp(torch.cat((h_emb, r_emb), dim=-1))
        loss = torch.sigmoid(torch.sum(t_emb * t_pred_emb))
        return loss


    def _build_cf_loss(self, users, items, labels):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        h_emb = self.entity_emb(items)
        for i in range(self.L):
            user_emb = self.user_mlp(user_emb)
            item_emb, h_emb = self.cc_unit(item_emb, h_emb)
        scores = self.cf_mlp(torch.cat((user_emb, item_emb), dim=-1))
        labels = labels.float().unsqueeze(1)
        loss = nn.BCEWithLogitsLoss()(scores, labels)
        return loss


    def _get_scores(self, users, items):
        user_emb = self.user_mlp(self.user_emb(users))
        for i in range(self.L):
            item_emb, _ = self.cc_unit(self.item_emb(items), self.entity_emb(items))
        scores = self.cf_mlp(torch.cat((user_emb, item_emb), dim=-1))
        scores = torch.sigmoid(scores)
        return scores


    def _parse_args(self, args, n_user, n_item, n_relation, n_entity):
        self.n_user = n_user
        self.n_item= n_item
        self.n_relation = n_relation
        self.n_entity = n_entity
        self.dim = args.dim
        self.L = args.L
        self.H = args.H
        self.l2_weight = args.l2_weight

    def linear_layer(self, in_dim, out_dim, dropout=0):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )


    def _build_higher_mlp(self, in_dim, out_dim, dropout=0):
        mlp = nn.Sequential() 
        for i in range(self.H - 1):
            mlp.add_module('dense{}'.format(i), nn.Linear(in_dim, in_dim))
            mlp.add_module('relu{}'.format(i), nn.LeakyReLU())
            mlp.add_module('drop{}'.format(i), nn.Dropout(dropout))
        mlp.add_module('dense', nn.Linear(in_dim, out_dim))
        return mlp


    def cc_unit(self, item_emb, h_emb):
        item_emb_reshape = item_emb.unsqueeze(-1)
        h_emb_reshape = h_emb.unsqueeze(-1)
        c = item_emb_reshape * h_emb_reshape.permute((0, 2, 1))
        c_t = h_emb_reshape * item_emb_reshape.permute((0, 2, 1))
        item_emb_c = torch.matmul(c,self.weight_vv).squeeze() + \
                       torch.matmul(c_t, self.weight_ev).squeeze() + self.bias_v 
        h_emb_c = torch.matmul(c, self.weight_ve).squeeze() + \
                       torch.matmul(c_t, self.weight_ee).squeeze() + self.bias_e
        return item_emb_c, h_emb_c


    def _init_weight(self):
        # init Embedding
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
    
    
    def forward(self, mode, *input):
        if mode == 'get_kge_loss':
            return self._build_kge_loss(*input)
        if mode == 'get_cf_loss':
            return self._build_cf_loss(*input)
        if mode == 'predict':
            return self._get_scores(*input)