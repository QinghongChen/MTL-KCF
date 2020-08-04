import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import CrossCompressUnit


class MODEL(nn.Module):
    def __init__(self, args, n_user, n_item, n_relation, n_entity):
        super(MODEL, self).__init__()
        self._parse_args(args, n_user, n_item, n_relation, n_entity)
        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.item_emb = nn.Embedding(self.n_item, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)

        self.user_mlp = self._build_lower_mlp(self.dim, self.dim)
        self.tail_mlp = self._build_lower_mlp(self.dim, self.dim)
        self.cf_mlp = self._build_higher_mlp(self.dim * 2, 1)
        self.kge_mlp = self._build_higher_mlp(self.dim * 2, self.dim)
        self.cc_unit = CrossCompressUnit(self.dim)

        self._init_weight()

    def _build_kge_loss(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        _, h_emb = self.cc_unit(self.item_emb(h), self.entity_emb(h))
        t_emb = self.tail_mlp(t_emb)
        t_pred_emb = self.kge_mlp(torch.cat((h_emb, r_emb), dim=-1))
        loss = torch.mean(torch.sum(torch.square(t_emb - t_pred_emb), dim=1), dim=0)
        print(loss)
        return loss


    def _build_cf_loss(self, users, items, labels):
        user_emb = self.user_mlp(self.user_emb(users))
        item_emb, _ = self.cc_unit(self.item_emb(items), self.entity_emb(items))
        scores = self.cf_mlp(torch.cat((user_emb, item_emb), dim=-1))
        labels = labels.float().unsqueeze(1)
        loss = nn.BCELoss()(scores, labels)
        print(loss)
        return loss


    def _get_scores(self, users, items):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        scores = self.cf_mlp(torch.cat((user_emb, item_emb), dim=-1))
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


    def _build_lower_mlp(self, in_dim, out_dim):
        mlp = nn.Sequential()
        for i in range(self.L):
            mlp.add_module('dense{}'.format(i), nn.Linear(in_dim, out_dim, bias=True))
            mlp.add_module('relu{}'.format(i), nn.ReLU())
        return mlp

    def _build_higher_mlp(self, in_dim, out_dim):
        mlp = nn.Sequential() 
        for i in range(self.H - 1):
            mlp.add_module('dense{}'.format(i), nn.Linear(in_dim, in_dim, bias=True))
            mlp.add_module('relu{}'.format(i), nn.ReLU())
        mlp.add_module('dense', nn.Linear(in_dim, out_dim, bias=True))
        mlp.add_module('sigmoid', nn.Sigmoid())
        return mlp


    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        # init mlp
        for mlp in [self.user_mlp, self.tail_mlp]:
            for layer in mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
    
    
    def forward(self, mode, *input):
        if mode == 'get_kge_loss':
            return self._build_kge_loss(*input)
        if mode == 'get_cf_loss':
            return self._build_cf_loss(*input)
        if mode == 'predict':
            return self._get_scores(*input)