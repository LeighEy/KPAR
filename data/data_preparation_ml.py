import pandas as pd
import pickle
import argparse
from collections import defaultdict
import random
from tqdm import tqdm
import sys
from os import path, mkdir
import os
from pathlib import Path
import KPAN.constants.consts as consts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--movies_file',
                        default='ml_imdb.csv',
                        help='Path to the CSV file containing song information')
    parser.add_argument('--interactions_file',
                        default='interactions.csv',
                        help='Path to the CSV file containing user song interactions')
    parser.add_argument('--subnetwork',
                        default='dense',
                        choices=['dense', 'rs', 'sparse', 'full'],
                        help='The type of subnetwork to form from the full KG')
    # Additional arguments (following bug)
    parser.add_argument('--mode',
                        default='client')
    parser.add_argument('--port',
                        default=51131)
    return parser.parse_args()


def make_person_list(row):
    if row['person_list'].split(',')[0] == '':
        return list()
    else:
        return row['person_list'].split(',')


def movies_data_prep(movies_file, interactions_file, export_dir):
    '''
    :return: Write out 4 python dictionaries for the edges of KG
    '''

    movies = pd.read_csv(movies_file,index_col=0)
    interactions = pd.read_csv(interactions_file,index_col=0)

    # item_person.dict
    # dict where key = item_id, value = list of persons (artists, composers, lyricists)
    person = movies[['movie_id', 'imdb_name_id']].drop_duplicates()
    movie_person = person.fillna('').groupby('movie_id')['imdb_name_id'].apply(lambda x: ','.join(x)).reset_index()
    movie_person.columns = ['movie_id', 'person_list']
    movie_person['person_list'] = movie_person.apply(lambda x: make_person_list(x), axis=1)
    # movie_person = pd.concat([movies['song_id'], person_list], axis=1)

    movie_person_dict = movie_person.set_index('movie_id')['person_list'].to_dict()
    with open(os.path.join(export_dir, consts.ITEM_PERSON_DICT), 'wb') as handle:
        pickle.dump(movie_person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # person_item.dict
    # dict where key = a person, value = list of items related to this person
    person_movie_dict = {}
    for row in movie_person.iterrows():
        for person in row[1]['person_list']:
            if person not in person_movie_dict:
                person_movie_dict[person] = []
            person_movie_dict[person].append(row[1]['movie_id'])
    with open(os.path.join(export_dir, consts.PERSON_ITEM_DICT), 'wb') as handle:
        pickle.dump(person_movie_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # item_user.dict
    # dict where key = movie_id, value = list of user_ids
    movie_user = interactions[['movie_id', 'user_id']].set_index('movie_id').groupby('movie_id')['user_id'].apply(list).to_dict()
    with open(os.path.join(export_dir, consts.ITEM_USER_DICT), 'wb') as handle:
        pickle.dump(movie_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_item.dict
    # dict where key = user_id, value = list of movies_ids
    user_song = interactions[['user_id', 'movie_id']].set_index('user_id').groupby('user_id')['movie_id'].apply(list).to_dict()
    with open(os.path.join(export_dir, consts.USER_ITEM_DICT), 'wb') as handle:
        pickle.dump(user_song, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # user_song_tuple.txt
    # numpy array of [user_id, song_id] pairs sorted in the order of user_id
    user_song_tuple = interactions[['user_id', 'movie_id']].sort_values(by='user_id').to_string(header=False, index=False,
                                                                                         index_names=False).split('\n')
    user_song_tuple = [row.split() for row in user_song_tuple]
    with open(os.path.join(export_dir,'user_movie_tuple.txt'), 'wb') as handle:
        pickle.dump(user_song_tuple, handle, protocol=pickle.HIGHEST_PROTOCOL)


def find_subnetwork(network_type, dir, factor=0.1):
    if network_type == 'full':
        return

    # Load Data

    with open(dir + consts.ITEM_USER_DICT, 'rb') as handle:
        movie_user = pickle.load(handle)
    with open(dir + consts.USER_ITEM_DICT, 'rb') as handle:
        user_movie = pickle.load(handle)
    with open(dir + consts.ITEM_PERSON_DICT, 'rb') as handle:
        movie_person = pickle.load(handle)
    with open(dir + consts.PERSON_ITEM_DICT, 'rb') as handle:
        person_movie = pickle.load(handle)
    movie_user = defaultdict(list, movie_user)
    movie_person = defaultdict(list, movie_person)
    user_movie = defaultdict(list, user_movie)
    person_movie = defaultdict(list, person_movie)

    # Sort Nodes By Degree in descending order

    # key: song, value: number of users listening to it + number of person relating to its creation
    movie_degree_dict = {}
    for (k, v) in movie_user.items():
        movie_degree_dict[k] = v
    for (k, v) in movie_person.items():
        # if the song is already exist in dict, update the value (relations) by adding relevant persons
        if k in movie_degree_dict.keys():
            movie_degree_dict[k] = movie_degree_dict[k] + v
        else:
            movie_degree_dict[k] = v
    movie_degree = [(k, len(v)) for (k, v) in movie_degree_dict.items()]
    movie_degree.sort(key=lambda x: -x[1])

    # key: person, value: number of songs they create
    person_degree = [(k, len(v)) for (k, v) in person_movie.items()]
    person_degree.sort(key=lambda x: -x[1])

    # key: user, value: number of songs they listen to
    user_degree = [(k, len(v)) for (k, v) in user_movie.items()]
    user_degree.sort(key=lambda x: -x[1])

    # Construct Subnetworks

    # find the nodes
    print('finding the nodes...')
    # if network_type == 'dense':
    if 'dense' in network_type:
        if 'top' in network_type:
            start = 10
        else:
            start = 0
        movie_nodes_holder = movie_degree[start:int(len(
            movie_degree) * factor)]  # song_id is the first item in the tuple element of the returned list
        movie_nodes = [node_holder[0] for node_holder in movie_nodes_holder]

        user_nodes_holder = user_degree[start:int(len(user_degree) * factor)]
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = person_degree[start:int(len(person_degree) * factor)]
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    elif network_type == 'rs':
        movie_nodes_holder = random.sample(movie_degree, int(len(
            movie_degree) * factor))  # song_id is the first item in the tuple element of the returned list
        movie_nodes = [node_holder[0] for node_holder in movie_nodes_holder]

        user_nodes_holder = random.sample(user_degree, int(len(user_degree) * factor))
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = random.sample(person_degree, int(len(person_degree) * factor))
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    elif network_type == 'sparse':
        movie_nodes_holder = movie_degree[-int(len(
            movie_degree) * factor):]  # song_id is the first item in the tuple element of the returned list
        movie_nodes = [node_holder[0] for node_holder in movie_nodes_holder]

        user_nodes_holder = user_degree[-int(len(user_degree) * factor):]
        user_nodes = [node_holder[0] for node_holder in user_nodes_holder]

        person_nodes_holder = person_degree[-int(len(person_degree) * factor):]
        person_nodes = [node_holder[0] for node_holder in person_nodes_holder]

    nodes = movie_nodes + user_nodes + person_nodes
    print(f'The {network_type} subnetwork has {len(nodes)} nodes: {len(movie_nodes)} songs, {len(user_nodes)} users, {len(person_nodes)} persons.')

    # find the edges
    # (node1, node2) and (node2, node1) both exist
    edges_type1 = []  # a list of pairs (item, user)
    edges_type2 = []  # a list of pairs (item, person)
    edges_type3 = []  # a list of pairs (user, item)
    edges_type4 = []  # a list of pairs (person, item)
    nodes_set = set(nodes)

    # for each node in the nodes set, check the intersection with their info in the datasets
    for i in tqdm(nodes_set):  # (node1, node2) and (node2, node1) both exist
        connect_1 = set(movie_user[i]).intersection(nodes_set) # item-user
        for j in connect_1:
            edges_type1.append((i, j))

        connect_2 = set(movie_person[i]).intersection(nodes_set) # item-person
        for j in connect_2:
            edges_type2.append((i, j))

        connect_3 = set(user_movie[i]).intersection(nodes_set) # user-item
        for j in connect_3:
            edges_type3.append((i, j))

        connect_4 = set(person_movie[i]).intersection(nodes_set) # person-movie
        for j in connect_4:
            edges_type4.append((i, j))

    edges = edges_type1 + edges_type2 + edges_type3 + edges_type4
    print(f'The {network_type} subnetwork has {len(edges)} edges.')

    # Export the Subnetworks

    # <NETWORK_TYPE>_song_user.dict
    # key: song, value: a list of users
    movie_user_dict = defaultdict(list)
    for edge in edges_type1:
        song = edge[0]
        user = edge[1]
        movie_user_dict[song].append(user)
    movie_user_dict = dict(movie_user_dict)
    prefix = dir + network_type + '_'
    with open(prefix + consts.ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(movie_user_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_song_person.dict
    # key: song, value: a list of persons
    movie_person_dict = defaultdict(list)
    for edge in edges_type2:
        song = edge[0]
        person = edge[1]
        movie_person_dict[song].append(person)
    movie_person_dict = dict(movie_person_dict)
    with open(prefix + consts.ITEM_PERSON_DICT, 'wb') as handle:
        pickle.dump(movie_person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_user_song.dict
    # key: user, value: a list of songs
    user_movie_dict = defaultdict(list)
    for edge in edges_type3:
        user = edge[0]
        song = edge[1]
        user_movie_dict[user].append(song)
    user_movie_dict = dict(user_movie_dict)
    with open(prefix + consts.USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(user_movie_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # <NETWORK TYPE>_person_song.dict
    # key: person, value: a list of songs
    person_movie_dict = defaultdict(list)
    for edge in edges_type4:
        person = edge[0]
        song = edge[1]
        person_movie_dict[person].append(song)
    person_movie_dict = dict(person_movie_dict)
    with open(prefix + consts.PERSON_ITEM_DICT, 'wb') as handle:
        pickle.dump(person_movie_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def convert_to_ids(entity_to_ix, rel_dict, start_type, end_type):
    new_rel = {}
    for key, values in rel_dict.items():
        key_id = entity_to_ix[(key, start_type)]
        value_ids = []
        for val in values:
            value_ids.append(entity_to_ix[(val, end_type)])
        new_rel[key_id] = value_ids
    return new_rel

def ix_mapping(network_type, import_dir, export_dir, mapping_export_dir):
    pad_token = consts.PAD_TOKEN
    type_to_ix = {'person': consts.PERSON_TYPE, 'user': consts.USER_TYPE, 'movie': consts.ITEM_TYPE,
                  pad_token: consts.PAD_TYPE}
    relation_to_ix = {'movie_person': consts.ITEM_PERSON_REL, 'person_movie': consts.PERSON_ITEM_REL,
                      'user_movie': consts.USER_ITEM_REL, 'movie_user': consts.ITEM_USER_REL, '#UNK_RELATION': consts.UNK_REL,
                      '#END_RELATION': consts.END_REL, pad_token: consts.PAD_REL}

    # entity vocab set is combination of songs, users, and persons
    if network_type == 'full':
        movie_data_prefix = import_dir
    else:
        movie_data_prefix = import_dir + network_type + '_'
    with open(movie_data_prefix + consts.ITEM_USER_DICT, 'rb') as handle:
        movie_user = pickle.load(handle)
    with open(movie_data_prefix + consts.ITEM_PERSON_DICT, 'rb') as handle:
        movie_person = pickle.load(handle)
    with open(movie_data_prefix + consts.USER_ITEM_DICT, 'rb') as handle:
        user_movie = pickle.load(handle)
    with open(movie_data_prefix + consts.PERSON_ITEM_DICT, 'rb') as handle:
        person_movie = pickle.load(handle)

    # unique nodes
    movies = set(movie_user.keys()) | set(movie_person.keys())
    users = set(user_movie.keys())
    persons = set(person_movie.keys())

    # Id-ix mappings
    entity_to_ix = {(movie, consts.ITEM_TYPE): ix for ix, movie in enumerate(movies)}
    entity_to_ix.update({(user, consts.USER_TYPE): ix + len(movies) for ix, user in enumerate(users)})
    entity_to_ix.update(
        {(person, consts.PERSON_TYPE): ix + len(movies) + len(users) for ix, person in enumerate(persons)})
    entity_to_ix[pad_token] = len(entity_to_ix)

    # Ix-id mappings
    ix_to_type = {v: k for k, v in type_to_ix.items()}
    ix_to_relation = {v: k for k, v in relation_to_ix.items()}
    ix_to_entity = {v: k for k, v in entity_to_ix.items()}

    # Export mappings
    movie_ix_mapping_prefix = mapping_export_dir + network_type + '_'
    # eg. song_ix_data/dense_type_to_ix.dict
    with open(movie_ix_mapping_prefix + consts.TYPE_TO_IX, 'wb') as handle:
        pickle.dump(type_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.RELATION_TO_IX, 'wb') as handle:
        pickle.dump(relation_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.ENTITY_TO_IX, 'wb') as handle:
        pickle.dump(entity_to_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.IX_TO_TYPE, 'wb') as handle:
        pickle.dump(ix_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.IX_TO_RELATION, 'wb') as handle:
        pickle.dump(ix_to_relation, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(movie_ix_mapping_prefix + consts.IX_TO_ENTITY, 'wb') as handle:
        pickle.dump(ix_to_entity, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Update the KG
    movie_user_ix = convert_to_ids(entity_to_ix=entity_to_ix, rel_dict=movie_user, start_type=consts.ITEM_TYPE, end_type=consts.USER_TYPE)
    user_movie_ix = convert_to_ids(entity_to_ix, user_movie, consts.USER_TYPE, consts.ITEM_TYPE)
    movie_person_ix = convert_to_ids(entity_to_ix, movie_person, consts.ITEM_TYPE, consts.PERSON_TYPE)
    person_movie_ix = convert_to_ids(entity_to_ix, person_movie, consts.PERSON_TYPE, consts.ITEM_TYPE)

    # export
    # eg. song_ix_data/dense_ix_song_user.dict
    ix_prefix = export_dir + network_type + '_ix_'
    with open(ix_prefix + consts.ITEM_USER_DICT, 'wb') as handle:
        pickle.dump(movie_user_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.USER_ITEM_DICT, 'wb') as handle:
        pickle.dump(user_movie_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.ITEM_PERSON_DICT, 'wb') as handle:
        pickle.dump(movie_person_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(ix_prefix + consts.PERSON_ITEM_DICT, 'wb') as handle:
        pickle.dump(person_movie_ix, handle, protocol=pickle.HIGHEST_PROTOCOL)


def train_test_split(network_type, dir):
    with open(dir + network_type + '_ix_' + consts.USER_ITEM_DICT, 'rb') as handle:
        user_movie = pickle.load(handle)

    # KG and positive
    train_user_movie = {}
    test_user_movie = {}
    train_movie_user = defaultdict(list)
    test_movie_user = defaultdict(list)

    for user in user_movie:
        pos_movies = user_movie[user]
        random.shuffle(pos_movies)
        cut = int(len(pos_movies) * 0.8)

        # train
        train_user_movie[user] = pos_movies[:cut]
        for song in pos_movies[:cut]:
            train_movie_user[song].append(user)

        # test
        test_user_movie[user] = pos_movies[cut:]
        for song in pos_movies[cut:]:
            test_movie_user[song].append(user)

    # Export
    # eg. song_ix_data/dense_train_ix_song_user.dict
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.USER_ITEM_DICT), 'wb') as handle:
        pickle.dump(train_user_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.USER_ITEM_DICT), 'wb') as handle:
        pickle.dump(test_user_movie, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_train_ix_%s' % (dir, network_type, consts.ITEM_USER_DICT), 'wb') as handle:
        pickle.dump(train_movie_user, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('%s%s_test_ix_%s' % (dir, network_type, consts.ITEM_USER_DICT), 'wb') as handle:
        pickle.dump(test_movie_user, handle, protocol=pickle.HIGHEST_PROTOCOL)

def create_directory(dir):
    print("Creating directory %s" % dir)
    try:
        mkdir(dir)
    except FileExistsError:
        print("Directory already exists")

def main():
    print("Data preparation:")
    args = parse_args()
    network_prefix = args.subnetwork

    print("Forming knowledge graph...")
    # update the location
    DATA_LOC = os.path.join(str(Path(os.getcwd()).parents[2]), 'Data', consts.DATASET_DOMAIN)
    PROCESSED_DATA = os.path.join(DATA_LOC,'processed_data')
    create_directory(os.path.join(PROCESSED_DATA, consts.ITEM_DATA_DIR))

    # define current locations
    movies_file_loc = os.path.join(DATA_LOC, consts.ITEM_DATASET_DIR, args.movies_file)
    interaction_file_loc = os.path.join(DATA_LOC, consts.ITEM_DATASET_DIR, args.interactions_file)
    movies_data_prep(movies_file=movies_file_loc, interactions_file=interaction_file_loc,
                     export_dir=os.path.join(PROCESSED_DATA, consts.ITEM_DATA_DIR))

    print("Forming network...")
    find_subnetwork(network_type=args.subnetwork, dir=os.path.join(PROCESSED_DATA, consts.ITEM_DATA_DIR))

    print("Mapping ids to indices...")
    create_directory(os.path.join(PROCESSED_DATA, consts.ITEM_IX_DATA_DIR))
    create_directory(os.path.join(PROCESSED_DATA, consts.ITEM_IX_MAPPING_DIR))
    ix_mapping(network_type=network_prefix,
               import_dir=os.path.join(PROCESSED_DATA, consts.ITEM_DATA_DIR),
               export_dir=os.path.join(PROCESSED_DATA, consts.ITEM_IX_DATA_DIR),
               mapping_export_dir=os.path.join(PROCESSED_DATA, consts.ITEM_IX_MAPPING_DIR))

    print("Creating training and testing datasets...")
    train_test_split(network_type=network_prefix,
                     dir=os.path.join(PROCESSED_DATA, consts.ITEM_IX_DATA_DIR))

    print('Data Prepartion is done!')

if __name__ == "__main__":
    main()