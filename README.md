# KPAN
We implemented the KPAN model on KKBox song dataset and MovieLens-IMDb (MI) dataset, compared it to a KPRN model (https://arxiv.org/abs/1811.04540). 
Our results are described in our paper.

Python version is 3.6.9, and environment requirements can be installed using `KPAN_requirements.yml`

## Usage Information
To train and evaluate the KPAN model, you can either use a random sample (subnetwork = rs), you can either use all data (subnetwork = full), or create a subnetwork yourself. 

The first step is download the data:
1) KKBox - download the `songs.csv` and `train.csv` from https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data. 
2) MI - download the `ratings.dat` and `movies.dat` from https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset, and from `movies.csv`, `names.csv` and `title.csv` from IMDb datasets (https://www.imdb.com/). Merge both datasets by movie ID and year and named the file: `ml_imdb.csv`. 

For simplicity, we will demonstrate on the KKBox domain, but for MI domain you need to replace the word 'song' in 'movie'.
Create a folder called `song_dataset` in `{root}/data` and place `songs.csv` and `train.csv` in `song_dataset`.

Then construct the knowledge graph with data_preparation.py (and data_preparation_ml.py for MI), and path-find, train, and evaluate using recommender.py.

### Knowledge Graph Construction
Run data_preparation.py/data_preparation_ml.py to create relevant dictionaries from the datasets.

Arguments:

`--songs_file`/`--movies_file` to specify path to CSV containing song/movie information

`--interactions_file` to specify path to CSV containing user-item interactions

`--subnetwork` to specify data to create knowledge graph from. Options are dense, rs, sparse, and full.
Dense contains the top 10% entities with highest degree, and rs contains a random 10% sample of entities. For our evaluation we use full.


### recommender.py command line arguments

`--train` to train model, `--validation` to add validation. `--eval` to evaluate

`--find_paths` if you want to find paths before training or evaluating

`--subnetwork` to specify subnetwork training and evaluating on.

`--model` designates the model to train or evaluate from

`--model_name` designates the specific model to train or evaluate: KPAN or KPRN

`--path_agg_methos` designates the way of path aggregation: attention (for cross attention) or weighted pooling

`--load_checkpoint` if you want to load a model checkpoint (weights and parameters) before training

`--kg_path_file` designates the file to save/load train/test paths from

`--user_limit` designates the max number of train/test users to find paths for

`-b` designates model batch size and `-e` number of epochs to train model for

`--not_in_memory` if training on entire dense subnetwork, whose paths cannot fit in memory all at once

`--lr`, `--l2_reg` specify model hyperparameters (learning rate, l2 regularization)

`--nhead`,`--dropout` specify hyperparameters for transformer layer

`--path_nhead` specify number of heads in path aggregation

`--entity_agg` designates the method for aggregate paths

`--random_samples` designates if the paths sampling will be random 

`--item-to-item` True for inference task of item-to-item similarity
