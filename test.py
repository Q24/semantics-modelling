import cPickle
import logging
import hred_vhred.dialog_encdec
import model_manager
from data import encoding_tools, word_embedding_tools
import dill
import numpy
from model_manager import ModelManager
from data.data_file import FileArray
from data.data_access import build_database_from_scratch, get_label_translator
from data.encoding_tools import save_embeddings_to_file, check_embeddings_consistency, encode
from ann.candidate_selection import *
import evaluation
#-0.133256561537

def chat_with_model(model_manager):
    from hred_vhred import search
    encoder = model_manager.load_currently_selected_model()
    sampler = search.BeamSampler(encoder)
    print encoder.state['reset_utterance_decoder_at_end_of_utterance']
    while 1:

        input = raw_input('context :')

        input += ' __eou__ </s>'
        samples, costs = sampler.sample([input.split()], n_samples=5, n_turns=1)
        for idx, sample in enumerate(samples[0]):
            print str(costs[0][idx])+ ': ' + sample

def convert_model(m):
    encoder = m.load_currently_selected_model()
    encoder.state['dictionary'] = m.folders['binarized']+'dict.pkl'
    with open(m.folders['model_versions']+'2016-12-23_20:15:49_40.897_model.pkl', 'wb') as f:
        dill.dump(encoder, f, protocol=dill.HIGHEST_PROTOCOL)

    m.save_state(encoder.state)

if __name__ == '__main__':
    #58.0168834256
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")



    m = ModelManager('ubuntu_vhred_vanilla')
    '''
    result_arr = FileArray('./results/context_and_answer_relevance_ubuntu_vhred_vanilla.bin', dtype='i4')
    result_arr.open()
    x = 0
    while 1:

        r = result_arr.read(x)[0]
        x += 1
        print r
        if r == 0:
            break
    result_arr.close()
    #'''
    #convert_model(m)

    #chat_with_model(m)
    #evaluation.evaluate_generative(m)
    evaluation.evaluate(m)
    evaluation.evaluate_lshf(m, context_and_answer_relevance)

    exit()

    from ann import lsh_forest
    from ann.candidate_selection import *

    m = ModelManager('ubuntu_vhred_vanilla')

    #from ann.lsh_forest import save_linked_utterance_embeddings
    #save_linked_utterance_embeddings(m)

    #lsh_forest.train_lsh_forest(m, corpus_percentage=0.05)
    ann = lsh_forest.load_lshf(m)
    utt_embs = lsh_forest.load_utterance_embeddings(m)
    encoder = m.load_currently_selected_model()

    embs = encode('how do i update all packages ? __eou__', encoder)

    d_emb = embs[0][0][0]

    distances, labels, embeddings = ann.kneighbors(d_emb, 10)

    translator = get_label_translator(m, as_text=True)

    labels = [(label[0], label[1]+1) for label in labels]

    search_context = {'distances':distances,
                      'labels':labels,
                      'candidate_dialogue_embeddings':embeddings,
                      'utterance_embeddings':utt_embs,
                      'original_utterance_embedding':embs[1][0][0],
                      'original_dialogue_embedding':embs[0][0][0]}
    scored = answer_relevance(search_context)
    scored = sorted(scored, key=lambda tpl: tpl[0])
    for score, label in scored:
        print score, translator(label)
        print

    exit()


    m = ModelManager('ubuntu_vhred_vanilla')

    #check_embeddings_consistency(m)
    save_embeddings_to_file(m)
    #build_database_from_scratch(m)
    exit()
    encoder = m.load_currently_selected_model()
    encoder.state['dictionary'] = m.folders['binarized']+'dict.pkl'
    with open(m.folders['current_version']+'2016-12-28_07:19:58_43.184_model.pkl', 'wb') as f:
        dill.dump(encoder, f, protocol=dill.HIGHEST_PROTOCOL)

    m.save_state(encoder.state)



    exit()

    m = model_manager.ModelManager('ubuntu_vhred_vanilla')

    encoder = m.load_currently_selected_model()

    print encoding_tools.create_model_specific_encoding_hash(encoder)

    with open(m.folders['model_versions']+'2016-12-28_07:19:58_43.184_model.pkl', 'wb') as f:
        dill.dump(encoder, f, protocol=dill.HIGHEST_PROTOCOL)

    exit()

    #with open('../models/test/model_versions/2017-06-01_09:51:00_inf_model.pkl', 'rb') as f:
    with open('./models/test/model_versions/2017-06-09_12:06:14_inf_model.pkl', 'rb') as f:
        model = cPickle.load(f)

    test_dialogue = '<customer> hoi </u> </s>'

    embeddings = encoding_tools.encode(test_dialogue, model)

    #with open('../models/test/model_versions/copy.pkl', 'wb') as f:
    #    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print len(embeddings[0][0][0])
    print len(embeddings[1][0][0])

    #print model.encoding_hash
    print encoding_tools.create_model_specific_encoding_hash(model)


    #with open('./models/test/model_versions/copy.pkl', 'wb') as f:
        #cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print model.timings['encoding_hash']