import os
import urllib2
import subprocess
import socket

from nltk import word_tokenize
from model_manager import ModelManager
from spell_correction import get_spell_corrector
from utils import get_script_dir
from preprocessing.anonymization import Anonymizer

PORT = 8000

STUFF_TO_DOWNLOAD = [('https://github.com/alxbar/dutch_private_infos/raw/master/names_postcodes.tar.gz', 'names_postcodes.tar.gz'),]

def initialize():
    data_folder = get_script_dir() + '/preprocessing'

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    for url, file_name in STUFF_TO_DOWNLOAD:
        save_path = data_folder + '/' + file_name

        if not os.path.exists(save_path):
            print '...downloading', url

            subprocess.call(['wget', '-O', save_path, url])

        if file_name.endswith('.tar.gz') and os.path.exists(save_path):
            print '...unpacking'
            subprocess.call(('tar -zxf /home/alex/chatbot/preprocessing/names_postcodes.tar.gz -C %s'%data_folder).split())
            os.remove(save_path)




'''
Pipeline that takes raw or almost raw sentences as input
and processes it to be optimally used by the HRED model.
'''
class FilterPipeline():

    def __init__(self, model_manager, language='dutch'):
        self.model_manager = model_manager
        self.language = language
        self.pipeline = []


    def run_server(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = "localhost"
        self.port = PORT
        self.s.bind((self.host, self.port))

        self.s.listen(5)
        print ('server started and listening')
        while 1:
            (clientsocket, address) = self.s.accept()
            data = clientsocket.recv(1024).decode()
            output = self.process(data)
            clientsocket.send(output.encode())



    def request_filtering(self, some_sentence):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('localhost', PORT))
        s.send(some_sentence.encode())
        data = s.recv(1024).decode()
        s.close()
        return data

    def process(self, inp):

        for processing_fn in self.pipeline:
            inp = processing_fn(inp)

        return inp

    def add(self, filter):
        self.pipeline.append(filter)


    def add_tokenizer(self, mind_meta_tokens=True):

        def tokenize(inp):
            tokens = word_tokenize(inp, language=self.language)

            if mind_meta_tokens:
                # find beginning and ending of meta tokens
                opening_indices = [idx for idx, string in enumerate(tokens) if string == '<']
                closing_indices = [idx for idx, string in enumerate(tokens) if string == '>']

                # sort start and end of meta-token-characters, remember position/idx of character
                combined_indices = sorted([(idx, True) for idx in opening_indices] + [(idx, False) for idx in closing_indices], key= lambda pair: pair[0])
                # retrieve only proper start and ending indices of meta-tokens
                combined_indices = [(el[0], combined_indices[idx+1][0]) for idx,el in enumerate(combined_indices[:-1]) if combined_indices[idx][1] and not combined_indices[idx+1][1]]
                # rearrange
                opening_indices = [opening for opening, _ in combined_indices]
                closing_indices = [closing for _, closing in combined_indices]

                processed_tokens = []

                composed_token = None



                # concatenate characters that compose meta tokens. Normal append for regular tokens
                for idx, token in enumerate(tokens):

                    if idx in opening_indices:
                        composed_token = '<'
                        continue

                    if idx in closing_indices:
                        processed_tokens.append(composed_token+'>')
                        composed_token = None
                        continue

                    if composed_token:
                        composed_token += token
                    else:
                        processed_tokens.append(token)

                tokens = processed_tokens

            return tokens

        self.pipeline.append(tokenize)



    def add_anonymization(self):
        an = Anonymizer()

        def anonymize_smth(inp):
            return an.anonymize([' '.join(inp)])
        self.pipeline.append(anonymize_smth)

    def add_spell_corrector(self):

        dictionary = self.model_manager.load_vocabulary(clean=True)
        corrector = get_spell_corrector(self.model_manager)

        def correct(inp):

            for sentence in inp:
                processed = []
                for token in sentence:

                    if token not in dictionary:
                        token = corrector(token)

                    if token not in dictionary:
                        token = '<unk>'

                    processed.append(token)

            return processed

        self.pipeline.append(correct)

    def add_finalizer(self, default_start = '<customer>'):

        def finalize(inp):
            if type(inp) in (list, tuple):
                inp = (' '.join(inp)).strip()

            if not inp.endswith('</u> </s>'):
                inp += ' </u> </s>'

            first_token = inp.split()[0].strip()

            if not first_token in ('<customer>','<assistant>'):
                inp = default_start +' ' +inp

            return inp

        self.pipeline.append(finalize)


if __name__ == '__main__':
    #initialize()




    m = ModelManager('vodafone_hred_v3')
    p = FilterPipeline(m)

    if 'y' in raw_input('client?(y/n)'):
        p.request_filtering(raw_input('Gimme    '))
        exit()

    print 'preparing filtering steps'
    p.add_tokenizer()
    p.add_anonymization()
    p.add_spell_corrector()
    p.add_finalizer()

    print 'starting server'
    p.run_server()


    #print p.process('<alex><hoik ikq ben alex! hm112123 asdjjwber <unk> </u> </s>')