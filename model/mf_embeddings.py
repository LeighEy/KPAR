import pickle
import pandas as pd
from sklearn.decomposition import NMF
from scipy import sparse
import KPAN.constants.consts as consts
import os
from pathlib import Path


class MatrixFactorization:

    def __init__(self, data_ix_path):
        self.data = self.prep_from_dict(data_ix_path)
        self.users_embeddings = None
        self.items_embeddings = None
        self.users_mapping = None
        self.items_mapping = None

    def prep_from_dict(self, data_ix_path, train_or_test='train'):
        with open(data_ix_path + f'_{train_or_test}_ix_{consts.ITEM_USER_DICT}', 'rb') as handle:
            item_user_train = pickle.load(handle)

        # keys are items and values are list of items
        user_item_pairs = [(i, v) for v, u in item_user_train.items() for i in u]
        # create df of user-item
        data = pd.DataFrame(user_item_pairs, columns=['user', 'item'])
        data['interact'] = 1
        return data

    def create_embeddings(self, num_of_components):
        # mapping the ids
        self.items_mapping = self.data['item'].drop_duplicates().reset_index(drop=True).reset_index().rename(
            columns={'index': 'item_idx'})
        self.users_mapping = self.data['user'].drop_duplicates().reset_index(drop=True).reset_index().rename(
            columns={'index': 'user_idx'})

        # the interactions are binary. so there is no meaning for sum/exp smoothing
        user_items_matrix = self.matrix_create()
        self.users_embeddings, self.items_embeddings = self.factorize_mat(user_items_matrix, num_of_components)

    def matrix_create(self):
        self.data = pd.merge(self.data, self.users_mapping, on='user')
        self.data = pd.merge(self.data, self.items_mapping, on='item')

        user_item_matrix = sparse.coo_matrix(
            (self.data['interact'], (self.data['user_idx'], self.data['item_idx'])),
            shape=(len(self.users_mapping), len(self.items_mapping)))

        return user_item_matrix

    def factorize_mat(self,matrix, num_of_components):
        nmf = NMF(num_of_components, verbose=False)
        W = nmf.fit_transform(matrix)
        H = nmf.components_

        return W, H

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "MatrixFactorization"
        return super().find_class(module, name)

if __name__ == "__main__":
    data_root = r'/home/hananb/Leigh'
    PROCESSED_DATA_DIR = os.path.join(data_root, 'Data', consts.DATASET_DOMAIN,
                                      'processed_data')
    data_ix_path = os.path.join(PROCESSED_DATA_DIR, consts.ITEM_IX_DATA_DIR + 'full')

    mf = MatrixFactorization(data_ix_path)
    print('create embeddings')
    mf.create_embeddings(num_of_components=consts.ENTITY_EMB_DIM)