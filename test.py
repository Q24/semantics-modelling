import cPickle
import logging
import hred_vhred.dialog_encdec
import model_manager
from data import encoding_tools, word_embedding_tools
import dill
import numpy
#-0.133256561537
if __name__ == '__main__':
    #58.0168834256
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    from model_manager import ModelManager
    from data.data_access import build_database_from_scratch
    from data.encoding_tools import save_embeddings_to_file, check_embeddings_consistency
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