import cPickle
import logging

from data import encoding_tools

#-0.133256561537
if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    #with open('../models/test/model_versions/2017-06-01_09:51:00_inf_model.pkl', 'rb') as f:
    with open('../models/test/model_versions/2017-06-01_10:30:02_inf_model.pkl', 'rb') as f:
        model = cPickle.load(f)

    test_dialogue = '<customer> hoi </u> </s>'

    embeddings = encoding_tools.encode(test_dialogue, model)

    #with open('../models/test/model_versions/copy.pkl', 'wb') as f:
    #    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print len(embeddings[0][0][0])
    print len(embeddings[1][0][0])

    #print model.encoding_hash
    for x in xrange(10):
        print encoding_tools.create_model_specific_encoding_hash(model)


    with open('../models/test/model_versions/copy.pkl', 'wb') as f:
        cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)

    print model.timings['encoding_hash']