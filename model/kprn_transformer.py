import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from KPAN.model.kprn import KPRN
# from KPRN.model.attention import MultiheadAttention
from torch.nn import MultiheadAttention
import KPAN.constants.consts as consts
import pickle
import os
import numpy as np


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_seq_len, 1, model_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x
        # return self.dropout(x)


class KPRNTransformer(KPRN):

    def __init__(self, e_emb_dim, t_emb_dim, r_emb_dim, hidden_dim, e_vocab_size, t_vocab_size, r_vocab_size,
                 tagset_size, no_rel, person_item_dict, dropout=0.1, nhead=8, nlayers=1, entities_agg='max',
                 init_mf_embedding=False, add_path_length=True, path_agg = 'weighted_pooling',path_nhead = 1,
                 mf_path = None):
        super(KPRNTransformer, self).__init__(e_emb_dim, t_emb_dim, r_emb_dim, hidden_dim, e_vocab_size,
                                              t_vocab_size, r_vocab_size, tagset_size, no_rel)

        # sum of embedding size (64+32+32=128)
        self.model_dim = e_emb_dim + t_emb_dim + r_emb_dim*(1-no_rel)
        # initialize PE for Self Attention
        self.pos_encoder = PositionalEncoding(model_dim=self.model_dim,dropout=dropout,max_seq_len=consts.MAX_PATH_LEN)
        encoder_layers = TransformerEncoderLayer(d_model=self.model_dim, nhead=nhead, dropout=dropout,
                                                 dim_feedforward=hidden_dim//8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.entity_agg = entities_agg

        if init_mf_embedding:
            self.initial_embeddings_mf(mf_path = mf_path,
                                       person_item=person_item_dict)

        # The linear layer that maps from hidden state space to tag
        if entities_agg == 'concat':
            concat_mul = 6*(consts.MAX_PATH_LEN == 6) + 4*(consts.MAX_PATH_LEN == 4)
        else:
            concat_mul = 1

        self.path_agg = path_agg
        self.linear1 = nn.Linear(self.model_dim*concat_mul, self.model_dim)
        self.linear2 = nn.Linear(self.model_dim, tagset_size)

        self.add_path_length = add_path_length
        if add_path_length: # add 1 for scalar for paths
            self.linear = nn.Linear(self.model_dim*concat_mul+1, tagset_size)
        else:
            self.linear = nn.Linear(self.model_dim * concat_mul, tagset_size)

        if path_agg == 'attention':
            # Attention to paths aggregation
            self.linear_score = nn.Linear(tagset_size,1)
            self.query_user_item_linear = nn.Linear(2 * consts.ENTITY_EMB_DIM, tagset_size)
            self.sigmoid = nn.Sigmoid()
            self.attention_model = MultiheadAttention(embed_dim=tagset_size,num_heads=1)

    def forward(self, paths, path_lengths, no_rel, num_of_paths=None):
        # transpose, so entities 1st row, types 2nd row, and relations 3rd (these are dim 1 and 2 since batch is 0)
        # this could just be the input if we want
        t_paths = torch.transpose(paths, 1, 2)

        # then concatenate embeddings, batch is index 0, so selecting along index 1
        # right now we do fetch embedding for padding tokens, but that these aren't used
        entity_embed = self.entity_embeddings(t_paths[:,0,:])
        type_embed = self.type_embeddings(t_paths[:,1,:])
        if no_rel:
            triplet_embed = torch.cat((entity_embed, type_embed), 2)  # concatenates lengthwise
        else:
            rel_embed = self.rel_embeddings(t_paths[:,2,:])
            triplet_embed = torch.cat((entity_embed, type_embed, rel_embed), 2) #concatenates lengthwise

        # we need dimensions to be input size x batch_size x embedding dim, so transpose first 2 dim
        batch_embed = torch.transpose(triplet_embed, 0, 1)
        batch_embed = batch_embed * math.sqrt(self.model_dim)
        batch_embed = self.pos_encoder(batch_embed)
        batch_embed = torch.transpose(batch_embed, 0, 1)
        output_trans = self.transformer_encoder(batch_embed)

        # new path representation
        if self.entity_agg == 'max':
            agg_output,_ = torch.max(output_trans,dim=1)
        elif self.entity_agg == 'mean':
            agg_output = output_trans.mean(dim=1)
        elif self.entity_agg == 'end_relation':
            len_6_idx = (path_lengths == 6).nonzero(as_tuple=False)
            len_4_idx = (path_lengths == 4).nonzero(as_tuple=False)
            len_6_representation = torch.squeeze(output_trans[len_6_idx, 5])
            len_4_representation = torch.squeeze(output_trans[len_4_idx, 3])
            if len(len_6_idx) == 1:
                len_6_representation = torch.unsqueeze(len_6_representation,dim=0)
            if len(len_4_idx) == 1:
                len_4_representation = torch.unsqueeze(len_4_representation,dim=0)
            agg_output = torch.cat((len_6_representation, len_4_representation), 0) # [X, 128]
        elif self.entity_agg == 'last': # representation of the last entity
            agg_output = output_trans[:,-1,:] # [X, 128]
        elif self.entity_agg == 'concat':
            out_size = output_trans.size()
            agg_output = torch.reshape(output_trans,(out_size[0],out_size[1]*out_size[2])) # [X,6*128=768]

        # pass through linear layers
        if self.add_path_length:
            print('Add path length as parameter')
            # reshape num of paths
            num_of_paths = torch.reshape(num_of_paths, (num_of_paths.shape[0], 1))
            tag_scores = self.linear(torch.cat((agg_output, num_of_paths), dim=1))
        else:
            # tag_scores = self.linear(agg_output)
            tag_scores = agg_output # remove FC after transformer

        if self.path_agg == 'weighted_pooling':
            tag_scores = self.linear2(self.linear1(agg_output))

        return tag_scores

    def initial_embeddings_mf(self, mf_path, person_item):
        if os.path.isfile(
                os.path.join(mf_path, 'mf_full_type_embeddings_as_array.pkl')):
            with open(os.path.join(mf_path,
                                   f'mf_full_embeddings_{consts.ENTITY_EMB_DIM}_as_array.pkl'), 'rb') as handle:
                embedding_as_array = pickle.load(handle)
            with open(os.path.join(mf_path,
                                   f'mf_full_type_embeddings_{consts.TYPE_EMB_DIM}_as_array.pkl'), 'rb') as handle:
                type_embedding_as_array = pickle.load(handle)
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_as_array), freeze=True)
            self.type_embeddings = nn.Embedding.from_pretrained(torch.tensor(type_embedding_as_array), freeze=True)

        else:

            with open(os.path.join(mf_path, f'mf_full_6040_{consts.ENTITY_EMB_DIM}_items_mapping.pkl'), 'rb') as handle:
                mf_items_mapping = pickle.load(handle)
            with open(os.path.join(mf_path, f'mf_full_6040_{consts.ENTITY_EMB_DIM}_users_mapping.pkl'), 'rb') as handle:
                mf_users_mapping = pickle.load(handle)
            with open(os.path.join(mf_path, f'mf_full_6040_{consts.ENTITY_EMB_DIM}_items_embeddings.pkl'), 'rb') as handle:
                mf_items_embeddings = pickle.load(handle).T
            with open(os.path.join(mf_path, f'mf_full_6040_{consts.ENTITY_EMB_DIM}_users_embeddings.pkl'), 'rb') as handle:
                mf_users_embeddings = pickle.load(handle)

            def update_embedding(df_mapping,df_embedding,embedding_to_update):
                mf_idx_nm = df_mapping.columns[0]
                cur_idx_nm = df_mapping.columns[1]

                for index, row in df_mapping.iterrows():
                    # assign the new values to the current embeddings
                    embedding_to_update[row[cur_idx_nm]] = df_embedding[row[mf_idx_nm]]

            # initialize the Embeddings as array
            embedding_as_array = np.zeros_like(np.array(self.entity_embeddings.weight.data))
            # update MF items embeddings
            update_embedding(df_mapping=mf_items_mapping,df_embedding=mf_items_embeddings,
                             embedding_to_update=embedding_as_array)
            # update MF user embeddings
            update_embedding(df_mapping=mf_users_mapping, df_embedding=mf_users_embeddings,
                             embedding_to_update=embedding_as_array)

            # initialize persons by Items
            for person in person_item.keys():
                embedding_as_array[person] = embedding_as_array[person_item[person]].mean(axis=0)
            with open(os.path.join(mf_path,
                                   f'mf_full_embeddings_{consts.ENTITY_EMB_DIM}_as_array.pkl'), 'wb') as handle:
                pickle.dump(embedding_as_array, handle)
            # update the embeddings and set for not update them
            self.entity_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_as_array),freeze=True)

            # # Update TYPE embeddings
            with open(os.path.join(mf_path, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_mapping.pkl'), 'rb') as handle:
                type_mf_items_mapping = pickle.load(handle)
            with open(os.path.join(mf_path, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_items_embeddings.pkl'), 'rb') as handle:
                type_mf_items_embeddings = pickle.load(handle).T
            with open(os.path.join(mf_path, f'mf_full_6040_comp_{consts.TYPE_EMB_DIM}_users_embeddings.pkl'), 'rb') as handle:
                type_mf_users_embeddings = pickle.load(handle)
            # type_embedding_as_array = np.array(self.type_embeddings.weight.data)
            type_embedding_as_array = np.zeros_like(np.array(self.type_embeddings.weight.data))
            # update item - mean over all items
            type_embedding_as_array[consts.ITEM_TYPE] = type_mf_items_embeddings.mean(axis=0)
            type_embedding_as_array[consts.USER_TYPE] = type_mf_users_embeddings.mean(axis=0)
            # update person
            shrink_person = np.zeros((len(person_item.keys()), type_mf_items_embeddings.shape[1]))
            for idx,person in enumerate(person_item.keys()):
                items_to_calc = []
                for item in person_item[person]:
                    if type_mf_items_mapping[type_mf_items_mapping.item == item]['item_idx'].shape[0] > 0:
                        items_to_calc.append(type_mf_items_mapping[type_mf_items_mapping.item == item]['item_idx'].values[0])
                if len(items_to_calc) == 0:
                    continue
                shrink_person[idx] = type_mf_items_embeddings[items_to_calc].mean(axis=0)
            type_embedding_as_array[consts.PERSON_TYPE] = shrink_person.mean(axis=0)
            # save the results
            with open(os.path.join(mf_path,
                                   f'mf_full_type_embeddings_{consts.TYPE_EMB_DIM}_as_array.pkl'), 'wb') as handle:
                pickle.dump(type_embedding_as_array, handle)
            self.type_embeddings = nn.Embedding.from_pretrained(torch.tensor(type_embedding_as_array),freeze=True)

    def calc_paths_attention(self, path_scores,user,item):
        paths_dim = torch.unsqueeze(path_scores, dim=0)
        # Define Q,K,V
        target_embed = torch.cat((self.entity_embeddings(user),self.entity_embeddings(item)))
        query = self.query_user_item_linear(target_embed)
        query_dim = query.unsqueeze(0).unsqueeze(0)
        key_value = torch.transpose(paths_dim, 0, 1)
        paths_att,weight_att = self.attention_model(query_dim,key_value,key_value,need_weights=True)

        score = self.sigmoid(torch.dot(query_dim.squeeze(), paths_att.squeeze()))
        return score,weight_att

    def embeddings_statistics(self,paths_embedding):
        min_stat = paths_embedding.min(dim=0)[0]
        max_stat = paths_embedding.max(dim=0)[0]
        median_stat = paths_embedding.median(dim=0)[0]
        paths_statistics = torch.cat((min_stat, max_stat, median_stat), dim=0)
        # run FC + sigmoid
        score = self.linear_score(paths_statistics)
        return self.sigmoid(score).squeeze(0)