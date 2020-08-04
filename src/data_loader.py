import collections
import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)


def load_data(args):
    logging.info("================== preparing data ===================")
    rating_data, n_user, n_item = load_rating(args)
    train_data, eval_data, test_data = dataset_split(rating_data)
    kg_data, n_relation, n_entity = load_kg(args)
    # train_data = pd.merge(train_data, kg_data, left_on='i', right_on='h')
    print_info(n_user, n_item, n_entity, n_relation)
    return train_data, eval_data, test_data, n_user, n_item, n_relation, n_entity, kg_data


def load_rating(args):
    rating_file = '../data/' + args.dataset + '/ratings_final.txt'
    logging.info("load rating file: %s", rating_file)
    rating_data = pd.read_csv(rating_file, sep='\t', names=['u', 'i', 'label'])
    n_user = max(rating_data['u']) + 1
    n_item = max(rating_data['i']) + 1
    return rating_data, n_user, n_item


def dataset_split(rating_data):
    logging.info("splitting dataset to 6:2:2 ...")
    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_data.shape[0]
    eval_indices = np.random.choice(n_ratings, size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    train_data = pd.DataFrame(rating_data.values[train_indices], columns=['u', 'i', 'label'])
    eval_data = pd.DataFrame(rating_data.values[eval_indices], columns=['u', 'i', 'label'])
    test_data = pd.DataFrame(rating_data.values[test_indices], columns=['u', 'i', 'label'])
    return train_data, eval_data, test_data


def load_kg(args):
    kg_file = '../data/' + args.dataset + '/kg_final.txt'
    logging.info("locading kg file: %s", kg_file)
    kg_data = pd.read_csv(kg_file, sep='\t', names=['h', 'r', 't'])
    kg_data = kg_data.drop_duplicates()
    n_relation = max(kg_data['r']) + 1
    n_entity = max(max(kg_data['h']), max(kg_data['t'])) + 1
    return kg_data, n_relation, n_entity


def print_info(n_user, n_item, n_entity, n_relation):
    print('n_user:            %d' % n_user)
    print('n_item:            %d' % n_item)
    print('n_entity:         %d' % n_entity)
    print('n_relation:        %d' % n_relation)