import numpy
import logging
import dill
import random
import os
from utils import print_progress_bar
from sklearn.neighbors import LSHForest
from gc import collect
from data.data_access import *
from data.data_file import FileArray
from time import time

class LSHForestPickable(LSHForest):

    def __setstate__(self, state):
        self.__dict__ = state

        for k, v in self.__dict__.iteritems():

            if str(type(v)) == "<type 'tuple'>":
                if len(v) == 3:
                    if str(v[0]) == '<built-in function _reconstruct>':
                        restored = v[0](v[1][0], v[1][1], v[1][2])
                        restored.__setstate__(v[2])

                        self.__dict__[k] = restored


    def __getstate__(self):

        for k, v in self.__dict__.iteritems():

            if str(type(v)) == "<type 'numpy.ndarray'>":

                self.__dict__[k] = None

        return self.__dict__

    def dump_data(self, file_loc):
        numpy_arrays = {}

        for k, v in self.__dict__.iteritems():

            if str(type(v)) == "<type 'numpy.ndarray'>":
                print 'found numpy array ', k
                numpy_arrays[k] = v
                self.__dict__[k] = None

        with open(file_loc, 'wb') as f:
            numpy.savez(f, **numpy_arrays)

    def load_data(self, file_loc):

        with open(file_loc, 'rb') as f:
            arrs = numpy.load(f)

            for variable_name in arrs.files:
                setattr(self, variable_name, arrs[variable_name])

        if hasattr(self, 'owner'):
            self.owner.embeddings = self._fit_X

        print



class EmbeddingsSpaceStructurizer():

    def __setstate__(self, d):
        self.__dict__ = d
        if type(self.nn) == LSHForestPickable:
            self.embeddings = self.nn._fit_X
            self.nn.owner = self
        else:
            self.embeddings = self.nn.data


    def __init__(self, data):

        # if data needs to be loaded
        if type(data)==str:
            data = self.load_embeddings_data(data)

        self.labels = data[1]

        logging.debug('... building nearest neighbour model')
        self.nn = LSHForestPickable(n_estimators=30, n_candidates=50).fit(data[0])

        del data


    def build_label_table(self):

        self.label_table = {}

        for index, (d_idx, turn) in enumerate(self.labels):

            if d_idx not in self.label_table:
                self.label_table[d_idx] = []

            self.label_table[d_idx].append((turn, index))

        print

        for k, v in self.label_table.iteritems():

            turn_indices = [pair[1] for pair in sorted(v, key= lambda p: p[0])]

            self.label_table[k] = turn_indices


    def nns_label_to_embedding(self, label):

        index = self.label_table[label[0]][label[1]]
        return self.embeddings[index]

    def kneighbors(self, embeddings_vector, n_neighbors=5):

        if type(self.nn) == LSHForestPickable:
            distances, indices = self.nn.kneighbors(X=[embeddings_vector], n_neighbors=n_neighbors)
        else:
            distances, indices = self.nn.query(x=[embeddings_vector], k=n_neighbors)


        embeddings = [self.embeddings[idx] for idx in indices[0]]
        labels = [self.labels[idx] for idx in indices[0]]

        return distances, labels, embeddings


def train_lsh_forest(model_manager, corpus_percentage = 0.30,  seed = 10):

    # get dialogue ids that can be used to train our ann model
    database = get_database(model_manager)
    train_ids = database[TRAIN_IDS_SET_NAME][:]

    # shuffle it to ensure that the training data is as versatile as possible
    logging.debug('shuffling training data with seed %i'%seed)
    rand = random.Random(seed)
    rand.shuffle(train_ids)

    # divide the data to ensure that everything fits into memory
    train_ids = train_ids[:(len(train_ids)*corpus_percentage)]
    logging.debug('will train LSH-Fores using %i conversations (%.2f%% of the corpus)'%(len(train_ids), 100.*len(train_ids)/len(database[TRAIN_IDS_SET_NAME])))


    # collect coords of embeddings related to our training indices
    coord_set = database[EMBEDDINGS_COORDINATES_SET_NAME]
    coords = [(coord_set[d_idx], d_idx) for d_idx in train_ids]



    # collect actual embeddings
    dia_embs = FileArray(model_manager.files['dialogue_embeddings'])
    embeddings = []
    labels = []

    logging.debug('...collecting embeddings for training')
    dia_embs.open()
    progress = 0
    start_time = time()
    for (global_idx, conv_lengh), d_idx in coords:
        progress += 1

        # read in embeddings of entire conversation
        embeddings.extend(dia_embs.read_chunk(global_idx, conv_lengh))

        # save corresponding dialogue_id and turn
        labels.extend([(d_idx, turn_idx) for turn_idx in range(conv_lengh)])

        if progress % 100 == 0:
            print_progress_bar(progress, len(coords), additional_text='%i embeddings gathered' % len(embeddings),
                               start_time=start_time)
    dia_embs.close()

    logging.debug('done collecting embeddings (%i gathered)' % len(embeddings))
    logging.debug('...training nearest neighbor model')
    nns = EmbeddingsSpaceStructurizer((embeddings, labels))

    logging.debug('training done')
    file_loc = model_manager.folders['nearest_neighbor']+'dia.lshf.pkl'
    arrays_loc = model_manager.folders['nearest_neighbor']+'dia.lshf.npz'

    logging.debug('saving embeddings of model')
    nns.nn.dump_data(arrays_loc)

    with open(file_loc, 'wb') as f:
        logging.debug('saving model')
        dill.dump(nns, f, protocol=dill.HIGHEST_PROTOCOL)

    collect()

def save_linked_utterance_embeddings(model_manager):
    with open(model_manager.folders['nearest_neighbor']+'dia.lshf.pkl') as f:
        model = dill.load(f)

    logging.debug('loading %i utterence embeddings'%len(model.labels))
    embeddings = load_embeddings(model_manager, model.labels)

    logging.debug('...saving')
    with open(model_manager.folders['nearest_neighbor']+'utt.embs.pkl', 'wb') as f:
        dill.dump(embeddings, f, protocol=dill.HIGHEST_PROTOCOL)


def load_lshf(model_manager):
    nns_loc = model_manager.folders['nearest_neighbor'] + 'dia.lshf.pkl'
    with open(nns_loc, 'rb') as f:
        nns = dill.load(f)

    nns.nn.load_data(nns_loc.replace('.pkl', '.npz'))
    nns.build_label_table()

    return nns

def load_utterance_embeddings(model_manager):

    with open(model_manager.folders['nearest_neighbor']+'utt.embs.pkl', 'rb') as f:
        embs = dill.load(f)

    return embs