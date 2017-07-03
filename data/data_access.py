import h5py
import logging
import numpy as np
from logging import debug
from time import time
from utils import print_progress_bar

DICTIONARY_NAME = 'dictionary'
BINARIZED_SET_NAME = 'dialogues_binarized'
TEXTUAL_SET_NAME = 'dialogues'
UTTERANCE_EMBEDDINGS_SET_NAME = 'utterance_embeddings'
DIALOGUE_EMBEDDINGS_SET_NAME = 'dialogue_embeddings'
EMBEDDINGS_COORDINATES_SET_NAME = 'embedding_coordinates'

TRAIN_IDS_SET_NAME = 'train_ids'
VALID_IDS_SET_NAME = 'valid_ids'
TEST_IDS_SET_NAME = 'test_ids'

CLUSTER_IDS_SET_NAME = 'cluster_ids'
RANDOM_CLUSTERS_SET_NAME = 'random_id_clusters'

POS_TAGS_SET_NAME = 'pos_tags'


def get_database(model_manager):
    database_loc = model_manager.folders['data'] + 'storage.hdf5'
    f = h5py.File(database_loc, 'a', libver='latest')
    return f

def create_dataset(model_manager, set_name, size):
    f = get_database(model_manager)
    dt = h5py.special_dtype(vlen=unicode)
    if set_name not in f:
        dset = f.create_dataset(set_name, (size,), dtype=dt)
    else:
        dset = f[set_name]
    return dset

def del_if_exists(database, dataset_name):
    if dataset_name in database:
        del database[dataset_name]
        return True
    return False

def build_database_from_scratch(model_manager):
    database = get_database(model_manager)
    debug('...loading dictionary')
    dictionary = model_manager.load_vocabulary()

    dict_set = create_dataset(model_manager=model_manager, set_name=DICTIONARY_NAME, size=len(dictionary))

    debug('...storing dictionary with size=%i'%len(dictionary))

    for word, word_id, _, _ in dictionary:
        dict_set[word_id] = word

    debug('dict done')

    logging.debug('...loading binarized dialogues')
    train_bin, valid_bin, test_bin = model_manager.load_data()
    train_size = len(train_bin)
    valid_size = len(valid_bin)
    test_size = len(test_bin)
    total_size = train_size+valid_size+test_size
    debug('loaded %i train_set, %i valid_set, %i test_set'%(train_size, valid_size, test_size))
    debug('total size:%i' % (train_size+valid_size+test_size))

    logging.debug('...storing binarized dialogues')
    del_if_exists(database, BINARIZED_SET_NAME)

    dt = h5py.special_dtype(vlen=np.dtype('int32'))
    dialogue_set = database.create_dataset(BINARIZED_SET_NAME, (total_size,), dtype=dt)

    dialogue_id = 0

    start_time = time()

    for set in [train_bin, valid_bin, test_bin]:
        for dialogue_indices in set:
            dialogue_set[dialogue_id] = dialogue_indices
            dialogue_id += 1

            print_progress_bar(dialogue_id+1, total_size, additional_text='%i binarized dialogues stored'%(dialogue_id+1),start_time=start_time)

    logging.debug('binarized done')

    logging.debug('...storing %i train ids'%train_size)
    del_if_exists(database, TRAIN_IDS_SET_NAME)

    train_ids_set = database.create_dataset(TRAIN_IDS_SET_NAME, (train_size,), dtype='i4')
    for word_id in range(train_size):
        train_ids_set[word_id] = word_id

    logging.debug('...storing %i valid ids' % valid_size)
    del_if_exists(database, VALID_IDS_SET_NAME)

    valid_ids_set = database.create_dataset(VALID_IDS_SET_NAME, (valid_size,), dtype='i4')
    for idx, word_id in enumerate(range(train_size, train_size+valid_size)):
        valid_ids_set[idx] = word_id

    logging.debug('...storing %i test ids' % test_size)
    del_if_exists(database, TEST_IDS_SET_NAME)

    test_ids_set = database.create_dataset(TEST_IDS_SET_NAME, (test_size,), dtype='i4')
    for idx, word_id in enumerate(range(train_size+valid_size, train_size+valid_size+test_size)):
        test_ids_set[idx] = word_id

    logging.debug('ids done')



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

