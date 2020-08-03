import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MODEL(nn.Module):
    def __init__(self, args, n_users, n_items, n_relations, n_entities):
        super(MODEL, self).__init__()
        self._parse_args(args, n_users, n_items, n_relations, n_entities)
        self.user_emb = nn.Embedding(self.n_user, self.dim)
        self.item_emb = nn.Embedding(self.n_item, self.dim)
        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.entity_emb = nn.Embedding(self.n_entity, self.dim)
        # self.W_R = nn.Parameter(torch.Tensor(self.n_relation, self.dim, self.dim))
        self.mlp_t = self._build_mlp(self.dim * 2, self.dim, False)
        self._init_weight()


    def calc_t_loss(self, h, r, t):
        h_emb = self.entity_emb(h)
        r_emb = self.relation_emb(r)
        t_emb = self.entity_emb(t)
        t_hat_emb = self.mlp_t(torch.cat((h_emb, r_emb), dim=-1))
        t_loss = torch.mean(torch.abs(torch.cosine_similarity(t_emb, t_hat_emb, dim=1)), dim=0)
        return t_loss


    def calc_it_loss(self, user, item):       
        item_emb = self.item_emb(item)
        h_emb = self.entity_emb(item)
        it_loss = torch.mean(torch.abs(torch.cosine_similarity(item_emb, h_emb, dim=1)), dim=0)
        return it_loss
    
    
    def calc_score(self, user, item):       
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        scores = torch.sigmoid((user_emb * item_emb).sum(dim=1))
        return scores


    def _parse_args(self, args, n_users, n_items, n_relations, n_entities):
        self.n_user = n_users
        self.n_item= n_items
        self.n_relation = n_relations
        self.n_entity = n_entities
        self.dim = args.dim
        self.mlp_layer = args.mlp_layer
        self.l2_weight = args.l2_weight


    def _build_mlp(self, in_dim, out_dim, if_sigmoid):
        mlp = nn.Sequential() 
        for i in range(self.mlp_layer - 1):
            mlp.add_module('dense{}'.format(i), nn.Linear(in_dim, in_dim, bias=False))
            mlp.add_module('relu{}'.format(i), nn.ReLU())
        mlp.add_module('out', nn.Linear(in_dim, out_dim, bias=False))
        if if_sigmoid:
            mlp.add_module('sigmoid', nn.Sigmoid())
        return mlp


    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        # init mlp
        for layer in self.mlp_t:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    
    def forward(self, mode, *input):
        if mode == 'calc_t_loss':
            return self.calc_t_loss(*input)
        if mode == 'calc_score':
            return self.calc_score(*input)
        if mode == 'calc_it_loss':
            return self.calc_it_loss(*input)