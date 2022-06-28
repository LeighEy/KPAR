DATASET_DOMAIN = 'movielens'
# DATASET_DOMAIN = 'songs'

# SONGS
if 'songs' in DATASET_DOMAIN:
    ITEM_DATASET_DIR = 'song_dataset/'
    ITEM_DATA_DIR = 'song_data/'
    ITEM_IX_DATA_DIR = 'song_data_ix/'
    ITEM_IX_MAPPING_DIR = 'song_ix_mapping/'
    PATH_DATA_DIR = 'path_data/'

    PERSON_ITEM_DICT = 'person_song.dict'
    ITEM_PERSON_DICT = 'song_person.dict'
    USER_ITEM_DICT = 'user_song.dict'
    ITEM_USER_DICT = 'song_user.dict'

# MOVIES
else:
    ITEM_DATASET_DIR = 'ml_dataset/'
    ITEM_DATA_DIR = 'movie_data/'
    ITEM_IX_DATA_DIR = 'movie_data_ix/'
    ITEM_IX_MAPPING_DIR = 'movie_ix_mapping/'
    PATH_DATA_DIR = 'path_data/'

    PERSON_ITEM_DICT = 'person_movie.dict'
    ITEM_PERSON_DICT = 'movie_person.dict'
    USER_ITEM_DICT = 'user_movie.dict'
    ITEM_USER_DICT = 'movie_user.dict'

# COMMON
TYPE_TO_IX = 'type_to_ix.dict'
RELATION_TO_IX = 'relation_to_ix.dict'
ENTITY_TO_IX = 'entity_to_ix.dict'
IX_TO_TYPE = 'ix_to_type.dict'
IX_TO_RELATION = 'ix_to_relation.dict'
IX_TO_ENTITY = 'ix_to_entity.dict'

PAD_TOKEN = '#PAD_TOKEN'
ITEM_TYPE = 0
USER_TYPE = 1
PERSON_TYPE = 2
PAD_TYPE = 3

ITEM_PERSON_REL = 0
PERSON_ITEM_REL = 1
USER_ITEM_REL = 2
ITEM_USER_REL = 3
UNK_REL = 4
END_REL = 5
PAD_REL = 6

ENTITY_EMB_DIM = 16 #16 #64
TYPE_EMB_DIM = 8 #8 #32
REL_EMB_DIM = 8 #8 #32
HIDDEN_DIM = 256
TAG_SIZE = 32 #32 #2 #16

MAX_PATH_LEN = 4 #6 #4
NEG_SAMPLES_TRAIN = 4
NEG_SAMPLES_TEST = 100

LEN_3_BRANCH = 50
LEN_5_BRANCH_TRAIN = 6
LEN_5_BRANCH_TEST= 10
