import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from collections import defaultdict

from KPAN.model import my_collate, sort_batch
import KPAN.constants.consts as consts

class TestInteractionData(Dataset):
    def __init__(self, formatted_data):
        self.data = formatted_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def convert_to_etr(e_to_ix, t_to_ix, r_to_ix, path, length):
    '''
    Converts a path of ids back to the original input format
    -not used for anything right now but could be useful for visualization
    '''
    ix_to_t = {v: k for k, v in t_to_ix.items()}
    ix_to_r = {v: k for k, v in r_to_ix.items()}
    ix_to_e = {v: k for k, v in e_to_ix.items()}
    new_path = []
    for i,step in enumerate(path):
        if i == length:
            break
        new_path.append([ix_to_e[step[0].item()], ix_to_t[step[1].item()], ix_to_r[step[2].item()]])
    return new_path

def predict(model, formatted_data, batch_size, device, no_rel, gamma, test_num_paths=None):
    '''
    -outputs predicted scores for the input test data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    -Since we are evaluating we ignore the tag here
    '''

    prediction_scores = []
    interaction_data = TestInteractionData(formatted_data)
    #shuffle false since we want data to remain in order for comparison
    test_loader = DataLoader(dataset=interaction_data, collate_fn=my_collate, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (interaction_batch, _) in test_loader:
            #construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
            paths = []
            lengths = []
            inter_ids = []
            num_of_paths = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

                if test_num_paths:
                    num_of_paths.extend([test_num_paths[path[0][0],path[-1][0]] for i in range(len(interaction_paths))])

            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)
            if test_num_paths:
                num_of_paths = torch.tensor(num_of_paths, dtype=torch.float).to(device)

            # sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel,num_of_paths)

            # Get weighted pooling of scores over interaction id groups
            start = True
            for i in range(len(interaction_batch)):
                # get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)

                # weighted pooled scores for this interaction
                pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                if start:
                    # unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = pooled_score.unsqueeze(0)
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, pooled_score.unsqueeze(0)), dim=0)

            prediction_scores.extend(F.softmax(pooled_scores, dim=1))

    # just want positive scores currently
    pos_scores = []
    for tensor in prediction_scores:
        pos_scores.append(tensor.tolist()[1])
    return pos_scores

def predict_dataloader(model, data_loader, device, no_rel, gamma, baselines=True, path_aggregation='weighted_pooling',
                       eval_per_pose=False, item_to_item=False):
    all_results = []
    path_length_baseline_results = []
    popularity_baseline_results = []

    with torch.no_grad():
        for interaction_batch in data_loader:
            # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id
            paths = []
            lengths = []
            inter_ids = []
            targets = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths[0]:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths[0]))])
                targets.append(interaction_paths[1].item())

        # for paths, lengths, inter_id, target in data_loader:
            t_paths = torch.tensor(paths, dtype=torch.long)
            t_lengths = torch.tensor(lengths, dtype=torch.long)
            t_inter_ids = torch.tensor(inter_ids, dtype=torch.long)

            # sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(t_paths, t_inter_ids, t_lengths)
            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel)

            # weighted pooled scores for this interaction - tag scores is for 1 user-item
            # if eval_per_pose:
            prediction_scores = []
            hit_at_k_scores = defaultdict(list)
            ndcg_at_k_scores = defaultdict(list)

            start = True
            for i in range(len(interaction_batch)):
                # get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)
                user = s_path_batch[inter_idxs[0]][0][0]
                if item_to_item:
                    item = s_path_batch[inter_idxs[0]][-2][0]
                else:
                    item = s_path_batch[inter_idxs[0]][-1][0]

                if path_aggregation == 'weighted_pooling':
                    pooled_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma)

                    # unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = pooled_score.unsqueeze(0)
                    pred_score = F.softmax(pooled_scores, dim=1)[0][1].item()
                elif path_aggregation == 'attention':
                    att_score,attention_weights = model.calc_paths_attention(tag_scores[inter_idxs],
                                                                             user.to(device),
                                                                             item.to(device))
                    pred_score = att_score.unsqueeze(0)[0].item()

                all_results.extend([[user.item(), item.item(), pred_score, targets[i]]])
                if eval_per_pose:
                    prediction_scores.append((pred_score))

                if baselines:
                    path_length = s_path_batch[inter_idxs].shape[0] #len(paths)
                    path_length_baseline_results.extend([[user.item(), item.item(), path_length, targets[i]]])
                    popularity_baseline_results.extend([[user.item(), item.item(), targets[i]]])

    if baselines:
        return all_results,path_length_baseline_results,popularity_baseline_results,hit_at_k_scores,ndcg_at_k_scores
    return all_results