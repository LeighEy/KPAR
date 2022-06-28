import math
import pandas as pd
import numpy as np

def hit_at_k(ranked_tuples, k):
    '''
    Checks if the pos interaction occured in the top k scores
    '''
    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            return 1
    return 0

def precision_at_k(ranked_tuples, k):
    '''
    Checks if the pos interaction occured in the top k scores
    '''
    # I changed the way of calculation, as such it will be per user
    num_of_hits = 0
    for (score, tag) in ranked_tuples[:k]:
        if tag == 1:
            num_of_hits += 1
    return num_of_hits/k

def recall_at_k(ranked_tuples, k):
    total_test_pos = sum([x[1] for x in ranked_tuples])
    pred_pos = sum([x[1] for x in ranked_tuples[:k]])
    if total_test_pos == 0:
        return 0
    return pred_pos/total_test_pos

def ndcg_at_k_per_pos(ranked_tuples, k):
    '''
    Article on ndcg: http://ethen8181.github.io/machine-learning/recsys/2_implicit.html
    ndcg_k = DCG_k / IDCG_k
    Say i represents index of or tag=1 in the top k, then since only one contribution to summation
    DCG_k = rel_i / log(i+1)
    IDCG_k = rel_i / log(1+1) since the first item is best rank
    ndcg_k = log(2) / log(i+1)
    Note we use log(2) / log(i+2) since indexing from 0

    tag = true label
    '''
    for i,(score, tag) in enumerate(ranked_tuples[:k]):
        if tag == 1:
            return math.log2(2) / math.log2(i + 2)
    return 0

def ndcg_at_k(ranked_tuples, k):
    '''
    Article on ndcg: http://ethen8181.github.io/machine-learning/recsys/2_implicit.html
    ndcg_k = DCG_k / IDCG_k
    Say i represents index of or tag=1 in the top k, then since only one contribution to summation
    DCG_k = rel_i / log(i+1)

    tag = true label
    '''
    dcg = dcg_calculate(ranked_tuples[:k])
    s_ranked_tuples = sorted(ranked_tuples,key=lambda x: x[1],reverse=True)
    idcg = dcg_calculate(s_ranked_tuples[:k])

    if idcg == 0:
        return 0
    return dcg/idcg

def dcg_calculate(tuples):
    tags = np.asfarray([x[1] for x in tuples])
    if tags.size:
        # I changed from log2 to log (as exist above)
        return np.sum((2 ** tags - 1) / np.log2(np.arange(2, tags.size + 2)))
    else:
        return 0

def mpr(ranked_df):
    # sum(r_ui * percentile_rank) / sum(positive label)
    MPR = sum(ranked_df['label'] * ((ranked_df['rank'] - 1) / max(ranked_df['rank'] - 1))) / sum(ranked_df['label'])
    return MPR

def prep_for_evaluation(lst):
    df_results = pd.DataFrame(lst, columns=['user', 'item', 'y_pred', 'label'])
    # rank results by y_pred
    ranks = df_results.groupby('user')['y_pred'].rank(ascending=False, method='first')
    ranks.name = 'rank'
    return pd.concat([df_results, ranks], axis=1)

def calc_scores_per_user(df, max_k, model_nm, mpr_metric=False):
    ''' df is pandas Dataframe with columns: user, item, y_pred, label'''
    # initialize list which holds records: (user,k,hit@k,ndcg@k)
    scores_per_user = []
    mpr_per_user = pd.DataFrame(columns=['user','mpr'])

    # find top scores per user
    df.set_index('user', inplace=True)
    for user in list(df.index.unique()):
        cur_user = df[df.index == user].sort_values('rank', ascending=True)
        merged = list(zip(cur_user['y_pred'], cur_user['label']))
        user_hit = []
        precision = []
        recall = []
        user_ndcg = []
        for k in range(1, max_k + 1):
            # if k == 1:
            #     print(f'This user {user} have {sum(cur_user["label"])} positives')
            user_hit.append(hit_at_k(merged, k))
            user_ndcg.append(ndcg_at_k(merged, k))
            precision.append(precision_at_k(merged, k))
            recall.append(recall_at_k(merged, k))
        scores_per_user.extend(list(zip([model_nm] * max_k,
                                        [user] * max_k,
                                        range(1, max_k + 1),
                                        user_hit,
                                        user_ndcg,
                                        precision,
                                        recall)))
        mpr_per_user = mpr_per_user.append({'user':int(user), 'mpr':mpr(cur_user)},ignore_index=True)

    if mpr_metric:
        return pd.DataFrame(scores_per_user, columns=['model','user', 'rank', 'hit', 'ndcg', 'precision','recall']),mpr_per_user
    else:
        return pd.DataFrame(scores_per_user, columns=['model', 'user', 'rank', 'hit', 'ndcg', 'precision', 'recall'])

def aggregate_results(df,group_by):
    return df.groupby(group_by).agg({'hit': 'mean','ndcg': 'mean','precision': 'mean','recall': 'mean'}).reset_index()