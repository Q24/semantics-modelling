import cPickle
import model_manager
import logging
import numpy
from theano import config
from gensim.models import word2vec as w2v
from tqdm import tqdm
from tqdm import trange
import os

def train_embeddings(model_manager, vec_length, fix_pretrained=False):
    m = model_manager

    logging.debug('loading training data and vocab...')

    with open(m.files['train'], 'rb') as f:
        binarized_lines = cPickle.load(f)

    with open(m.files['dict'], 'rb') as f:
        vocabulary = cPickle.load(f)

    vocabulary = {entity[1]:entity[0] for entity in vocabulary}

    sentences = []

    logging.debug('preparing data for the training of word embeddings...')
    for word_id_list in binarized_lines:

        sentence = []

        for word_id in word_id_list:
            sentence.append(vocabulary[word_id])

        sentences.append(sentence)

    logging.debug('...training word2vec model (feature_length=%d)' % vec_length)

    model = w2v.Word2Vec(sentences, size=vec_length, min_count=0)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    logging.debug('Training done!')

    vocab = dict([(x[1], x[0]) for x in m.load_vocabulary()])
    vocab_size = len(vocab)

    logging.debug('vocab size %d' % vocab_size)
    logging.debug('vocab size of word2vec model %d' % len(model.vocab))



    rng = numpy.random.RandomState(0)

    values = numpy.zeros((vocab_size, vec_length), dtype=config.floatX)
    train_mask = numpy.zeros((vocab_size, vec_length), dtype=config.floatX)

    num_random_inits = 0
    logging.debug('...collecting embeddings')
    for word_id in range(vocab_size):
        word_str = vocab.get(word_id, 0)

        mask_row = numpy.ones((vec_length), dtype=config.floatX)
        if word_str in model:
            feature_vec = model[word_str]
            if fix_pretrained:
                mask_row = numpy.zeros((vec_length), dtype=config.floatX)

        else:
            feature_vec = rng.normal(loc=0, scale=0.01, size=(vec_length,))
            num_random_inits += 1

        train_mask[word_id] = mask_row
        values[word_id] = feature_vec

    logging.debug('number of randomly initialized word forms: %d' % num_random_inits)

    final_embeddings = []
    final_embeddings.append(values.astype(config.floatX))
    final_embeddings.append(train_mask.astype(config.floatX))

    logging.debug('...saving embeddings')
    with open(m.folders['word_embeddings'] + 'word2vec.embeddings.pkl', 'wb') as f:
        cPickle.dump(final_embeddings, f, protocol=cPickle.HIGHEST_PROTOCOL)


def load_pretrained_embeddings(model_manager, pretrained_loc, fix_pretrained=False):
    '''
    Given a certain dictionary and a model containing pre-trained word embeddings,
    this function searches for the embeddings that can be linked to the words in
    the dictionary, such that only the needed embeddings are stored.

    Words which are in the dictionary but do not have a corresponding word embedding
    in the pre-trained model are initialized randomly.
    '''
    with open(model_manager.files['dict'], 'rb') as f:
        vocabulary = cPickle.load(f)

    vocabulary = {entity[0]:entity[1] for entity in vocabulary}

    state = model_manager.load_current_state()

    rng = numpy.random.RandomState(0)

    feature_length = state['rankdim']

    values = numpy.zeros((len(vocabulary), feature_length), dtype=config.floatX)
    train_mask = numpy.ones((len(vocabulary), feature_length), dtype=config.floatX)

    logging.debug('Initializing word embeddings randomly')
    # initialize each word embedding randomly
    for word_str, word_id in vocabulary.iteritems():
        mask_row = numpy.ones((feature_length), dtype=config.floatX)
        feature_vec = rng.normal(loc=0, scale=0.01, size=(feature_length,))
        train_mask[word_id] = mask_row
        values[word_id] = feature_vec

    logging.debug('Iterating over word embeddings of pre-trained model')
    with open(pretrained_loc, 'rb') as f:
        file_name = os.path.splitext(os.path.basename(f.name))[0]
        header = f.readline()
        lines, vec_l = header.split(' ')
        lines = int(lines)
        vec_l = int(vec_l)
        assert feature_length == vec_l

        # overwrite word embeddings if found in pre-trained model

        pbar = tqdm(total=lines)

        found_matches = 0
        for line in f:
            word_end = line.find(' ')
            word = line[:word_end]
            pbar.update(1)
            if word not in vocabulary:
                continue

            found_matches += 1
            pbar.set_description('Found %i word embeddings'%found_matches)
            pbar.refresh()

            word_embedding = numpy.array([float(weight_str) for weight_str in line.split(' ')[1:]], dtype=config.floatX)
            word_id = vocabulary[word]

            values[word_id] = word_embedding
            if fix_pretrained:
                mask_row = numpy.zeros((feature_length), dtype=config.floatX)
                train_mask[word_id] = mask_row



    final_embeddings = []
    final_embeddings.append(values.astype(config.floatX))
    final_embeddings.append(train_mask.astype(config.floatX))

    logging.debug('...saving embeddings')
    with open(model_manager.folders['word_embeddings'] + file_name + '.embeddings.pkl', 'wb') as f:
        cPickle.dump(final_embeddings, f, protocol=cPickle.HIGHEST_PROTOCOL)











if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


