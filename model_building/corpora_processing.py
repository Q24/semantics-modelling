import logging
import cPickle
import collections
from collections import Counter
from utils import print_progress_bar

def convert_to_binarized_data(model_manager, source_file, target_dir, train_set_size=0.85, valid_set_size=0.10):
    m = model_manager

    logging.debug('loading textual data...')
    with open(source_file, 'rb') as f:
        lines = f.readlines()
    logging.debug('loaded %i lines'%len(lines))

    logging.debug('Creating the dictionary and converting the text to word-id lists')

    dictionary = {'<unk>':0, '</s>':1}

    word_id = 2

    binarized_corpus = []

    for progress, line in enumerate(lines):

        list_word_ids = []

        for word in line.strip().split():
            if word in dictionary:
                list_word_ids.append(dictionary[word])
                continue

            dictionary[word] = word_id
            list_word_ids.append(word_id)

            word_id += 1

        if progress%100 == 0:
            print_progress_bar(progress+1, len(lines))

        binarized_corpus.append(list_word_ids)

    print

    logging.debug('lines parsed, %i individual words form the dictionary'%(len(dictionary)))

    final_vocab = [(word, word_id, 0,0) for word, word_id in dictionary.iteritems()]

    valid_data_size = int((valid_set_size + train_set_size) * len(binarized_corpus))
    train_data_size = int(train_set_size * len(binarized_corpus))
    train_data = binarized_corpus[:train_data_size]

    valid_data = binarized_corpus[train_data_size:valid_data_size]

    test_data = binarized_corpus[valid_data_size:]

    logging.debug(
        "...saving training data (%d) to: %s" % (len(train_data), target_dir + m.model_name + ".train.dialogues.pkl"))

    cPickle.dump(train_data, open(target_dir + m.model_name + ".train.dialogues.pkl", 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)

    logging.debug(
        "...saving test data (%d) to: %s" % (len(test_data), target_dir + m.model_name + ".test.dialogues.pkl"))
    cPickle.dump(test_data, open(target_dir + m.model_name + ".test.dialogues.pkl", 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)

    logging.debug(
        "...saving valid data (%d) to: %s" % (len(valid_data), target_dir + m.model_name + ".valid.dialogues.pkl"))
    cPickle.dump(valid_data, open(target_dir + m.model_name + ".valid.dialogues.pkl", 'wb'),
                 protocol=cPickle.HIGHEST_PROTOCOL)

    logging.debug('...saving vocab (%d) to: %s' % (len(final_vocab), target_dir + m.model_name + ".dict.pkl"))
    cPickle.dump(final_vocab, open(target_dir + m.model_name + ".dict.pkl", 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def build_dictionary(lines, **kwargs):
    token_min_freq = kwargs.get('token_min_freq', 10)
    word_counter = Counter()
    logging.debug('...building dictionary')
    for idx, line in enumerate(lines):
        line_words = line.strip().split()
        if line_words[len(line_words) - 1] != '</s>':
            line_words.append('</s>')

        s = [x for x in line_words]
        word_counter.update(s)

    words_in_data = sum([word_counter[word] for word in word_counter])


    logging.debug('dictionary initial size: %d' % len(word_counter))
    logging.debug('initial number of words in data: %d'%words_in_data)
    word_counter = Counter(el for el in word_counter.elements() if word_counter[el] >= token_min_freq)

    words_after_filtering = sum([word_counter[word] for word in word_counter])

    logging.debug('dictionary size after filtering for min_freq=%d, size: %d' % (token_min_freq,len(word_counter)))
    logging.debug('number of words not represented anymore: %d (%.3f%%)'%(words_in_data-words_after_filtering, 100. * float(words_in_data-words_after_filtering)/words_in_data))
    return word_counter


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_set_size', type=float,
                        help='The size of the training set given by a floating point number between 0 and 1.',
                        default=0.85)
    parser.add_argument('-valid_set_size', type=float,
                        help='The size of the validation set given by a floating point number between 0 and 1.',
                        default=0.10)
    input = parser.parse_args()

    convert_to_binarized_data('test','/home/alex/chatbot/data/corpora/vodafone.lines.raw.txt','')
