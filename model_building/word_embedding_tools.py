import cPickle
import model_manager
import logging
import numpy
from theano import config
from gensim.models import word2vec as w2v

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



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    m = model_manager.ModelManager('test')
    train_embeddings(m,300)

