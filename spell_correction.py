'''
Taken from: https://github.com/mattalcock/blog/blob/master/2012/12/5/python-spell-checker.rst
which originated from: http://norvig.com/spell-correct.html
'''
import re, collections
import model_manager
import cPickle
import logging
import os

alphabet = 'abcdefghijklmnopqrstuvwxyz'




def train(model_manager, save=True):
    vocab = dict([(tok_id, tok) for tok, tok_id, _, _ in model_manager.load_vocabulary()])
    data = model_manager.load_train_data()

    frequencies = {}
    logging.debug('vocab size: %i, number of conversations: %i'%(len(vocab), len(data)))
    logging.debug('counting words...')

    for word_id_list in data:

        for word_id in word_id_list:
            word = vocab[word_id]

            if word not in frequencies:
                frequencies[word] = 1
            else:
                frequencies[word] += 1
    if(save):
        with open(model_manager.folders['nlp_models']+'frequencies.pkl', 'wb') as f:
            logging.debug('saving word frequencies to %s'%(model_manager.folders['nlp_models']+'frequencies.pkl'))
            cPickle.dump(frequencies, f, cPickle.HIGHEST_PROTOCOL)

    return frequencies




def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word, NWORDS):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words, NWORDS):
    return set(w for w in words if w in NWORDS)

def correct(word, NWORDS):
    candidates = known([word], NWORDS) or known(edits1(word), NWORDS) or known_edits2(word, NWORDS) or [word]
    return max(candidates, key=NWORDS.get)


def get_spell_corrector(model_manager):

    assert os.path.exists(model_manager.files['frequencies'])

    with open(model_manager.files['frequencies'], 'rb') as f:
        NWORDS = cPickle.load(f)

    def wrapper(some_word):
        return correct(some_word, NWORDS)

    return wrapper

if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    m = model_manager.ModelManager('test')

    corr = get_spell_corrector(m)



    print corr('hoik')
    print corr('smakelig')

    print corr('dankjeiwel')


