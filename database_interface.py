import logging
import h5py
import numpy as np
from datetime import time
from utils import print_progress_bar
from gc import collect

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

ALL_SET_NAMES = [DICTIONARY_NAME, BINARIZED_SET_NAME, TEXTUAL_SET_NAME, UTTERANCE_EMBEDDINGS_SET_NAME, DIALOGUE_EMBEDDINGS_SET_NAME,
                 EMBEDDINGS_COORDINATES_SET_NAME, TRAIN_IDS_SET_NAME, VALID_IDS_SET_NAME, TEST_IDS_SET_NAME, CLUSTER_IDS_SET_NAME, RANDOM_CLUSTERS_SET_NAME, POS_TAGS_SET_NAME]



def get_database(model_manager):
    database_loc = model_manager.folders['data'] + 'database.hdf5'
    f = h5py.File(database_loc, 'a')
    model_manager.index_important_files()
    return f

def create_dataset(model_manager, set_name, size):
    f = get_database(model_manager)
    dt = h5py.special_dtype(vlen=unicode)
    if set_name not in f:
        dset = f.create_dataset(set_name, (size,), dtype=dt)
    else:
        dset = f[set_name]
    return dset


def build_database_from_scratch(model_manager):
    '''
    Will initialize a hdf5 database for the given model.

    The textual and binarized data will be saved, as well as the dictionary.

    This database can later be filled with utterance and context embeddings.
    With a retrieval-based approach, similar conversations can be retrieved.

    The hdf5 data format ensures quick loading of data from the hard disk to RAM.
    '''
    database = get_database(model_manager)
    logging.debug('...loading dictionary')
    dictionary = model_manager.load_vocabulary()

    dict_set = create_dataset(model_manager=model_manager, set_name=DICTIONARY_NAME, size=len(dictionary))

    logging.debug('...storing dictionary with size=%i'%len(dictionary))

    for word, word_id in dictionary.iteritems():
        dict_set[word_id] = word

    logging.debug('dict done')

    logging.debug('...loading textual dialogues')
    train_set, valid_set, test_set = model_manager.load_data_textual()
    train_size = len(train_set)
    valid_size = len(valid_set)
    test_size = len(test_set)
    total_size = train_size+valid_size+test_size

    logging.debug('loaded %i train_set, %i valid_set, %i test_set'%(train_size, valid_size, test_size))
    logging.debug('total size:%i' % (train_size+valid_size+test_size))

    logging.debug('...storing textual dialogues')
    del_if_exists(database, TEXTUAL_SET_NAME)
    dt = h5py.special_dtype(vlen=unicode)
    dialogues_txt = database.create_dataset(TEXTUAL_SET_NAME, (total_size,), dtype=dt)

    dialogue_id = 0

    start_time = time()

    for set in [train_set, valid_set, test_set]:
        for dialogue in set:
            dialogues_txt[dialogue_id] = dialogue
            dialogue_id += 1
            if dialogue_id % 20 == 0:
                print_progress_bar(dialogue_id+1, total_size, additional_text='%i dialogues stored'%(dialogue_id+1),start_time=start_time)
    print
    logging.debug('textual dialogues done')

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

    logging.debug('...loading binarized dialogues')
    train_bin, valid_bin, test_bin = model_manager.load_data_binarized()

    logging.debug('...storing %i binarized dialogues'%(total_size))
    del_if_exists(database, BINARIZED_SET_NAME)

    dt = h5py.special_dtype(vlen=np.dtype('int32'))
    dialogue_set = database.create_dataset(BINARIZED_SET_NAME, (total_size,), dtype=dt)

    dialogue_id = 0

    start_time = time()

    for set in [train_bin, valid_bin, test_bin]:
        for dialogue_indices in set:
            dialogue_set[dialogue_id] = dialogue_indices
            dialogue_id += 1
            if dialogue_id % 20 == 0:
                print_progress_bar(dialogue_id+1, total_size, additional_text='%i binarized dialogues stored'%(dialogue_id+1),start_time=start_time)
    print

    logging.debug('binarized done')

def store_embeddings(model_manager, batch_size=50):

    encoder = model_manager.load_currently_selected_model()

    utt_features = encoder.state['qdim_encoder']
    dia_features = encoder.state['sdim']

    database = get_database(model_manager)

    dset = database[BINARIZED_SET_NAME]

    logging.debug('loading binarized dialogues into memory and storing index')
    binarized = [(d_idx, indices) for d_idx, indices in enumerate(dset)]


    logging.debug('...creating and storing mapping from (dialogue id and utterance id) to (embedding storage location)')

    del_if_exists(database, EMBEDDINGS_COORDINATES_SET_NAME)
    coords_set = database.create_dataset(EMBEDDINGS_COORDINATES_SET_NAME,  (len(binarized),2), dtype='i4')

    enlargened_idx = 0

    progress = 0
    start_time = time()

    for d_idx, indices in binarized:
        progress += 1

        conv_length = num_turns(indices, encoder.eos_sym)
        coords = (enlargened_idx, conv_length)

        coords_set[d_idx] = coords

        enlargened_idx += conv_length
        if progress % 100 == 0:
            print_progress_bar(progress, len(binarized), additional_text='coords for %i dialogues created'%progress, start_time=start_time)

    num_embeddings = enlargened_idx

    logging.debug('...sorting based on dialouge length for quicker encoding')
    dialouge_ids, binarized = zip(*sorted(binarized, key=lambda tuple: len(tuple[1]), reverse=True))


    logging.debug('...creating datasets for utterance and dialogue embeddings')

    del_if_exists(database, UTTERANCE_EMBEDDINGS_SET_NAME)
    del_if_exists(database, DIALOGUE_EMBEDDINGS_SET_NAME)

    utt_emb_set = database.create_dataset(UTTERANCE_EMBEDDINGS_SET_NAME, (num_embeddings, utt_features), dtype='f4')
    dia_emb_set = database.create_dataset(DIALOGUE_EMBEDDINGS_SET_NAME, (num_embeddings, dia_features), dtype='f4')

    logging.debug('...encoding %i dialouges'%len(binarized))
    collect()


    progress = 0
    batches_to_process = len(binarized)/batch_size
    start_time = time()

    total_embeddings_encoded = 0

    for d_indices, batch in zip(batch_iterator(dialouge_ids, batch_size=batch_size), batch_iterator(binarized, batch_size=batch_size, apply_on_element=lambda indices: np.append([encoder.eos_sym],np.append(indices, encoder.eos_sym)))):
        progress += 1


        embeddings = encode_batch_to_embeddings(encoder, batch)

        coords = [coords_set[d_idx] for d_idx in d_indices]

        for embs, coord in zip(embeddings, coords):
            utt_embs = embs[0]
            dia_embs = embs[1]

            assert len(utt_embs) == coord[1]
            assert len(dia_embs) == coord[1]

            assert len(utt_embs[0]) == utt_features
            assert len(dia_embs[0]) == dia_features

            for local_idx, global_idx in enumerate(xrange(coord[0], coord[0]+coord[1])):

                utt_emb_set[global_idx] = utt_embs[local_idx]
                dia_emb_set[global_idx] = dia_embs[local_idx]
                total_embeddings_encoded += 1

                if total_embeddings_encoded % 10000 == 0:
                    collect()



        print_progress_bar(progress, batches_to_process, additional_text='%i batches of dialogues processed (total of %i dialogues) (total of %i embeddings)'%(progress, ((progress-1)*batch_size)+len(batch), total_embeddings_encoded), start_time=start_time)

def num_turns(indices, eos_sym=1):
    counts = sum(1 for value in indices if value == eos_sym)
    if indices[0] == eos_sym:
        counts -= 1

    if not indices[-1] == eos_sym:
        counts += 1
    return counts

def del_if_exists(database, dataset_name):
    if dataset_name in database:
        del database[dataset_name]
        return True
    return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    from model_manager import ModelManager
    m = ModelManager('test')

    #build_database_from_scratch(m)
    store_embeddings(m)