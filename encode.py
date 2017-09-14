import sys, os
import argparse
import filter_pipeline
import logging
import numpy
from model_manager import ModelManager
from data import encoding_tools
from scipy.spatial.distance import cosine

class EncoderWrapper():

    def __init__(self, model, server_preprocessor = True):

        if type(model) in (basestring, str):
            model = ModelManager(model)

        self.model = model
        self.pipe = filter_pipeline.FilterPipeline(self.model)
        self.server_preprocessor = server_preprocessor
        self.encoder = self.model.load_currently_selected_model()

        if not self.server_preprocessor:
            self.pipe.add_tokenizer()
            self.pipe.add(lambda smth: [smth])
            self.pipe.add_spell_corrector()
            self.pipe.add_finalizer()

    def encode(self, conv):

        if self.server_preprocessor:
            conv = self.pipe.request_filtering(conv)
        else:
            conv = self.pipe.process(conv)

        print conv
        encoded = encoding_tools.encode(conv, self.encoder)

        for idx, arr in enumerate(encoded):
            for turn_idx, turn in enumerate(arr):
                encoded[idx][turn_idx] = encoded[idx][turn_idx][0]

        return encoded

    def cosine(self, emb1, emb2):
        return cosine(emb1, emb2)

def load_input(args):

    if os.path.exists(args.input):
        with open(args.input) as f:
            return f.readlines()


def encode(args):
    m = ModelManager(args.model_name)
    inp = load_input(args)

    wrapper = EncoderWrapper(m, args.preprocess == 'server')



    for conv in inp:


        encoded = wrapper.encode(conv)


        if args.return_type == 'print':
            print str(encoded)

        if args.return_type == 'npz':
            with open(args.result_save_loc, 'wb') as f:
                numpy.savez(f, numpy.array(encoded))
    #print



if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str,
                        help="how to return the encoded conversation", default='vodafone_hred_v3')
    parser.add_argument("-input", type=str,
                        help="how sentences are given")
    parser.add_argument("-return_type", type=str,
                        help="how to return the encoded conversation", default='npz')
    parser.add_argument("-result_save_loc", type=str,
                        help="where to store the embeddings", default='./embs.npz')
    parser.add_argument("-preprocess", type=str,
                        help="clean the sentences before sending to encoder", default='simple')
    args = parser.parse_args()

    encode(args)


















