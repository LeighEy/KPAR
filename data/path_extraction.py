import sys
from os import path
import pickle
import random
import KPAN.constants.consts as consts
from collections import defaultdict
import copy


class PathState:
    def __init__(self, path, length, entities):
        self.path = path    # array of [entity, entity type, relation to next] triplets
        self.length = length
        self.entities = entities    # set to keep track of the entities alr in the path to avoid cycles

def get_random_index(nums, max_length):
    index_list = list(range(max_length))
    random.shuffle(index_list)
    return index_list[:nums]


def find_paths_user_to_items(start_user, item_person, person_item, item_user, user_item, max_length, sample_nums):
    '''
    Finds sampled paths of max depth from a user to a sampling of songs
    '''
    item_to_paths = defaultdict(list)
    stack = []
    start = PathState([[start_user, consts.USER_TYPE, consts.END_REL]], 0, {start_user})
    stack.append(start)
    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        # add path to song_to_user_paths dict, just want paths of max_length rn since either length 3 or 5
        # if the path is over (by a song) - add to paths
        if type == consts.ITEM_TYPE and front.length == max_length:
            item_to_paths[entity].append(front.path)

        # limit of the path, without song at the end
        if front.length == max_length:
            continue

        # the first path starts from here, and also for other users entities
        if type == consts.USER_TYPE and entity in user_item:
            song_list = user_item[entity]
            # sample positive song based on indices
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                # if the song is not already in the path
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    # replace the last relation with the correct relation: user-song
                    new_path[-1][2] = consts.USER_ITEM_REL
                    new_path.append([song, consts.ITEM_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    stack.append(new_state)

        # if the node is song
        elif type == consts.ITEM_TYPE:
            # if this song exist in song for users
            if entity in item_user:
                user_list = item_user[entity]
                index_list = get_random_index(sample_nums, len(user_list))
                for index in index_list:
                    user = user_list[index]
                    if user not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_USER_REL
                        new_path.append([user, consts.USER_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{user})
                        stack.append(new_state)
            # if this song exist in song of persons
            if entity in item_person:
                person_list = item_person[entity]
                index_list = get_random_index(sample_nums, len(person_list))
                for index in index_list:
                    person = person_list[index]
                    if person not in front.entities:
                        new_path = copy.deepcopy(front.path)
                        new_path[-1][2] = consts.ITEM_PERSON_REL
                        new_path.append([person, consts.PERSON_TYPE, consts.END_REL])
                        new_state = PathState(new_path, front.length + 1, front.entities|{person})
                        stack.append(new_state)

        # if node is a person
        elif type == consts.PERSON_TYPE and entity in person_item:
            song_list = person_item[entity]
            index_list = get_random_index(sample_nums, len(song_list))
            for index in index_list:
                song = song_list[index]
                if song not in front.entities:
                    new_path = copy.deepcopy(front.path)
                    new_path[-1][2] = consts.PERSON_ITEM_REL
                    new_path.append([song, consts.ITEM_TYPE, consts.END_REL])
                    new_state = PathState(new_path, front.length + 1, front.entities|{song})
                    stack.append(new_state)

    return item_to_paths

def find_negative_paths_len_2(start_user, neg_list, max_length):
    '''
    Finds sampled paths of max depth from a user to a sampling of songs
    '''
    item_to_paths = defaultdict(list)
    stack = []
    for neg_item in neg_list:
        start = PathState([[start_user, consts.USER_TYPE, consts.USER_ITEM_REL]], 0, {start_user})
        new_path = copy.deepcopy(start.path)
        new_path.append([neg_item, consts.ITEM_TYPE, consts.END_REL])
        new_state = PathState(new_path, start.length + 1, start.entities | {neg_item})
        stack.append(new_state)

    while len(stack) > 0:
        front = stack.pop()
        entity, type = front.path[-1][0], front.path[-1][1]
        if type == consts.ITEM_TYPE and front.length == max_length:
            item_to_paths[entity].append(front.path)

    return item_to_paths

def main():
    with open("song_data_ix/dense_ix_song_person.dict", 'rb') as handle:
        song_person = pickle.load(handle)

    with open("song_data_ix/dense_ix_person_song.dict", 'rb') as handle:
        person_song = pickle.load(handle)

    with open("song_data_ix/dense_ix_song_user.dict", 'rb') as handle:
        song_user = pickle.load(handle)

    with open("song_data_ix/dense_ix_user_song.dict", 'rb') as handle:
        user_song = pickle.load(handle)

    print(find_paths_user_to_items(224218, song_person, person_song, song_user, user_song, 3, 1))


if __name__ == "__main__":
    main()
