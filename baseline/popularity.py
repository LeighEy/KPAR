import pickle
import argparse
import random
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from statistics import mean
import mmap
import numpy as np
import matplotlib.pyplot as plt
import os
from os import mkdir
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from pathlib import Path
import KPAN.constants.consts as consts
from KPAN.eval import precision_at_k, ndcg_at_k
from KPAN.eval import calc_scores_per_user, prep_for_evaluation, aggregate_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline',
                        type=str,
                        default='popularity',
                        help='name of the baseline')
    parser.add_argument('--kg_path_file',
                        type=str,
                        default='interactions.txt',
                        help='file name to store/load train/test paths')
    parser.add_argument('--subnetwork',
                        default= 'dense',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to load data from')
    parser.add_argument('--user_limit',
                        type=int,
                        default=5,
                        help='max number of users to find paths for')
    return parser.parse_args()

def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

class PopularityBL():
    def __init__(self):
        # self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
        self.items_score = None
        self.baseline_nm = 'popularity'

    def calculate_popularity(self, data_ix_path):
        # read the new file
        if consts.DATASET_DOMAIN == 'songs':
            user_limit = 10000
        else:
            user_limit = 6040
        with open(os.path.join(data_ix_path,'mf',f'processed_train_full_{user_limit}.pkl'), 'rb') as handle:
            processed_train = pickle.load(handle)
        # group by item and count
        item_rank = processed_train.groupby('item').size().reset_index().rename({0: 'y_pred'}, axis=1)
        self.items_score = item_rank[['item','y_pred']]

    def test_preparation(self,test_path,data_dir):

        file_path = os.path.join(data_dir, consts.PATH_DATA_DIR, test_path)

        users = []
        items = []
        labels = []
        set_path_length = []
        with open(file_path, 'r') as file:
            for line in tqdm(file, total=get_num_lines(file_path)):
                test_interactions = eval(line.rstrip("\n"))

                for path_obj in test_interactions:
                    actual_label = path_obj[1]
                    random_path = path_obj[0][0]
                    # len of the path to know where is the end item
                    len_path = random_path[1]
                    init_user = random_path[0][0][0]
                    end_item = random_path[0][len_path - 1][0]

                    # add to lists
                    users.append(init_user)
                    items.append(end_item)
                    labels.append(actual_label)
                    # set_path_length.append(len(path_obj[0]))

        processed_test = pd.DataFrame(list(zip(users, items, labels)),
                                      columns=['user', 'item', 'label'])
        return processed_test

    def predict(self,processed_test=pd.DataFrame(),data_dir=None,test_file_path=None):
        if processed_test.empty:
            processed_test = self.test_preparation(data_dir=data_dir,test_path=test_file_path)
        pop_predictions = processed_test.merge(self.items_score, on='item', how='left')
        pop_predictions['y_pred'] = pop_predictions['y_pred'].fillna(0)
        # re-ordering
        return pop_predictions[['user','item','y_pred','label']]

    def evaluate(self, predictions, args):
        merged = list(zip(predictions.y_pred,predictions.label))
        s_merged = sorted(merged, key=lambda x: x[0], reverse=True)

        hit_at_k_scores = defaultdict(list)
        ndcg_at_k_scores = defaultdict(list)
        max_k = 15
        for k in range(1, max_k + 1):
            hit_at_k_scores[k].append(precision_at_k(s_merged, k))
            ndcg_at_k_scores[k].append(ndcg_at_k(s_merged, k))

        scores = []

        for k in hit_at_k_scores.keys():
            hit_at_ks = hit_at_k_scores[k]
            ndcg_at_ks = ndcg_at_k_scores[k]
            print()
            print(["Average hit@K for k={0} is {1:.4f}".format(k, mean(hit_at_ks))])
            print(["Average ndcg@K for k={0} is {1:.4f}".format(k, mean(ndcg_at_ks))])
            scores.append([args.baseline, args.kg_path_file, k, mean(hit_at_ks), mean(ndcg_at_ks)])

        scores_cols = ['baseline', 'test_file', 'k', 'hit', 'ndcg']
        scores_df = pd.DataFrame(scores, columns=scores_cols)

        return scores_df