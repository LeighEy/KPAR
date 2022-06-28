import KPAN.constants.consts as consts
from tqdm import tqdm
import pickle
import os

def convert_train_paths_to_interactions(file_name):
    '''
    Converts train path file to list of (user,song) interaction tuples
    '''
    pos_interactions = []
    neg_interactions = []

    data_path = consts.PATH_DATA_DIR + file_name
    with open(data_path, 'r') as f:
        for line in f:
            interaction = eval(line.rstrip("\n"))
            marker = interaction[1]

            path_tuple = interaction[0][0]
            length = path_tuple[-1]
            user = path_tuple[0][0][0]
            song = path_tuple[0][length-1][0]

            if marker == 1:
                pos_interactions.append((user, song))
            elif marker == 0:
                neg_interactions.append((user, song))
            else:
                print("problem")

    return pos_interactions, neg_interactions


def convert_test_paths_to_interactions(file_name,processed_data_dir):
    '''
    Converts test path file to list of (user,song) interaction tuples
    '''
    pos_interactions = []
    neg_interactions = []

    data_path = os.path.join(processed_data_dir,
                             consts.PATH_DATA_DIR,file_name)
    with open(data_path, 'r') as f:
        for line in f:
            interactions = eval(line.rstrip("\n"))
            for interaction in interactions:
                marker = interaction[1]

                path_tuple = interaction[0][0]
                length = path_tuple[-1]
                user = path_tuple[0][0][0]
                song = path_tuple[0][length-1][0]

                if marker == 1:
                    pos_interactions.append((user, song))
                elif marker == 0:
                    neg_interactions.append((user, song))
                else:
                    print("problem")

    return pos_interactions, neg_interactions


def save_interactions(interactions,processed_data_dir,file_name):
    with open(os.path.join(processed_data_dir,file_name), 'wb') as f:
        pickle.dump(interactions, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    '''
    Used to convert paths to interaction tuples for use with Baseline
    This ensures we are evaluating on the same interactions across models
    '''

    test_pos_inters, test_neg_inters = convert_test_paths_to_interactions(
        "test_full_6040_samples10_interactions.txt",f'/home/hananb/Leigh/Data/{consts.DATASET_DOMAIN}/processed_data')
    save_interactions(test_pos_inters, f'/home/hananb/Leigh/Data/BPR/{consts.DATASET_DOMAIN}', 'full_test_pos_interactions.txt')



if __name__ == "__main__":
    main()
