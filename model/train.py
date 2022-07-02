import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import linecache
import os
import numpy as np
import random
import mmap

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean
from KPAN.model.validation import validate,ValidationData
import time
from datetime import timedelta


class TrainInteractionData(Dataset):
    '''
    Dataset that can either store all interaction data in memory or load it line
    by line when needed
    '''
    def __init__(self, train_path_file, in_memory=True):
        self.in_memory = in_memory
        self.file = train_path_file
        self.num_interactions = 0
        self.interactions = []
        if in_memory:
            # store all interaction in-memory
            with open(self.file, "r") as f:
                for line in f:
                    self.interactions.append(eval(line.rstrip("\n")))
            self.num_interactions = len(self.interactions)
        else:
            with open(self.file, "r") as f:
                for line in f:
                    self.num_interactions += 1

    def __getitem__(self, idx):
        # load the specific interaction either from memory or from file line
        if self.in_memory:
            return self.interactions[idx]
        else:
            line = linecache.getline(self.file, idx+1)
            return eval(line.rstrip("\n"))

    def __len__(self):
        return self.num_interactions

def my_collate(batch):
    '''
    Custom dataloader collate function since we have tuples of lists of paths
    '''
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def sort_batch(batch, indexes, lengths):
    '''
    sorts a batch of paths by path length, in decreasing order
    return: ordered paths, ordered object indices, ordered lengths
    '''
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    indexes_tensor = indexes[perm_idx]
    return seq_tensor, indexes_tensor, seq_lengths


def train(model, train_path_file, batch_size, epochs, model_path, load_checkpoint, not_in_memory, lr, l2_reg, gamma,
          no_rel, samples=-1, random_sample=True,model_name='KPAN',validation=False,validation_dataloader=None,
          path_aggregation='weighted_pooling'):
    '''
    -trains and outputs a model using the input data
    -formatted_data is a list of path lists, each of which consists of tuples of
    (path, tag, path_length), where the path is padded to ensure same overall length
    '''
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device is", device)
    model = model.to(device)

    if path_aggregation == 'weighted_pooling':
        loss_function = nn.NLLLoss()
    else:
        loss_function = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    if model_name == 'KPAN':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if load_checkpoint:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # DataLoader used for batches
    print('Read training data')
    interaction_data = TrainInteractionData(train_path_file, in_memory=not not_in_memory)
    print(f'Time read data: {timedelta(seconds=time.time() - start)}')
    print('prepare DataLoader')
    start_load = time.time()
    train_loader = DataLoader(dataset=interaction_data, collate_fn=my_collate, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    print(f'Time of loading the file: {timedelta(seconds=time.time()-start_load)}')

    epoch_loss = []
    epoch_val_loss = []
    for epoch in range(epochs):
        print("Epoch is:", epoch+1)
        model.train()
        start_epoch = time.time()

        # Allows learning embeddings after 5 epochs
        if model_name == 'KPAN' and epoch == 5:
            print('Unfreezing Embedding layer')
            model.entity_embeddings.weight.requires_grad = True
            model.type_embeddings.weight.requires_grad = True

        losses = []
        for interaction_batch_raw, targets in tqdm(train_loader): # have tqdm here when not on colab
            if samples == -1 or not random_sample:
                interaction_batch = interaction_batch_raw.copy()
                len_paths_interactions = [len(interaction) for interaction in interaction_batch]
            else:
                interaction_batch = []
                len_paths_interactions = []
                for interaction in interaction_batch_raw:
                    interaction_batch.append(random.sample(interaction, min(samples,len(interaction))))
                    len_paths_interactions.append(len(interaction))

            # construct tensor of all paths in batch, tensor of all lengths, and tensor of interaction id (the paths in every source-target)
            paths = []
            lengths = []
            inter_ids = []
            num_of_paths = []
            for inter_id, interaction_paths in enumerate(interaction_batch):
                for path, length in interaction_paths:
                    paths.append(path)
                    lengths.append(length)
                inter_ids.extend([inter_id for i in range(len(interaction_paths))])

            # convert to Tensor
            inter_ids = torch.tensor(inter_ids, dtype = torch.long)
            paths = torch.tensor(paths, dtype=torch.long)
            lengths = torch.tensor(lengths, dtype=torch.long)

            # sort based on path lengths, largest first, so that we can pack paths
            s_path_batch, s_inter_ids, s_lengths = sort_batch(paths, inter_ids, lengths)

            # Pytorch accumulates gradients, so we need to clear before each instance
            model.zero_grad()

            # Run the forward pass
            tag_scores = model(s_path_batch.to(device), s_lengths.to(device), no_rel, num_of_paths)

            # Get weighted pooling of scores over interaction id groups
            start = True
            for i in range(len(interaction_batch)):
                # get inds for this interaction
                inter_idxs = (s_inter_ids == i).nonzero().squeeze(1)

                # weighted pooled scores for this interaction
                if path_aggregation == 'weighted_pooling':
                    agg_score = model.weighted_pooling(tag_scores[inter_idxs], gamma=gamma) # [16]
                elif path_aggregation == 'attention': # attention
                    agg_score,_ = model.calc_paths_attention(tag_scores[inter_idxs],
                                                             s_path_batch[inter_idxs][0][0][0].to(device),
                                                             s_path_batch[inter_idxs][0][-1][0].to(device))

                if start:
                    # unsqueeze turns it into 2d tensor, so that we can concatenate along existing dim
                    pooled_scores = agg_score.unsqueeze(0) # [1,16]
                    start = not start
                else:
                    pooled_scores = torch.cat((pooled_scores, agg_score.unsqueeze(0)), dim=0)

            if path_aggregation == 'weighted_pooling':
                prediction_scores = F.log_softmax(pooled_scores, dim=1)
            else:
                prediction_scores = pooled_scores

            # Compute the loss, gradients, and update the parameters by calling .step()
            if path_aggregation == 'weighted_pooling':
                loss = loss_function(prediction_scores.to(device), targets.to(device))
            else:
                loss = loss_function(prediction_scores.squeeze(-1).to(device), targets.to(torch.float).to(device))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        if model_name == 'KPAN':
            scheduler.step()
        print(f'finish epoch: {timedelta(seconds=(time.time()) - start_epoch)}')

        # VALIDATION
        if validation:
            start_validation = time.time()
            total_validation_loss = 0.0
            model.eval()

            val_pred_scores,val_targets = validate(model, validation_dataloader, device, no_rel, gamma, path_aggregation)
            val_targets = torch.as_tensor(val_targets)
            if path_aggregation == 'weighted_pooling':
                val_loss = loss_function(val_pred_scores.to(device), val_targets.to(device))
            else:
                val_loss = loss_function(val_pred_scores.squeeze(-1).to(device), val_targets.to(torch.float).to(device))
            total_validation_loss += val_loss.item()
            print(f'finish validation: {timedelta(seconds=time.time() - start_validation)}')


        print("train loss is:", mean(losses))
        epoch_loss.append(mean(losses))
        if np.isnan(mean(losses)):
            print("loss is nan")
            break
        if validation:
            print("validation loss is:", total_validation_loss)
            epoch_val_loss.append(total_validation_loss)

        if validation:
            epoch_loss_for_count = epoch_val_loss.copy()
        else:
            epoch_loss_for_count = epoch_loss.copy()

        # Save model to disk
        if (epoch == 0) or (epoch_loss_for_count[-1] == min(epoch_loss_for_count)):
            print("Saving checkpoint to disk...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
              }, model_path)
        else:
            print('val epoch loss wasnt improved')

        # early stopping
        if (epoch > 4) and all([i > min(epoch_loss_for_count) for i in epoch_loss_for_count[-6:]]):
            print('Val loss is not improving for 5 iterations')
            break

    return model,epoch_loss,epoch_val_loss
