import argparse
import cPickle
import cmd
import logging
import os
import shutil
from tqdm import tqdm
import requests
import ann.lsh_forest as lsh_forest
import corpora_processing
from data import word_embedding_tools
from hred_vhred import train
from model_manager import ModelManager
from data.data_access import build_database_from_scratch
from data.encoding_tools import save_embeddings_to_file, check_embeddings_consistency
DOWNLOADS = [('dutch word embeddings: COW (COrpora from the Web)', 'http://www.clips.uantwerpen.be/dutchembeddings/cow-big.tar.gz', 'cow-big.tar.gz','./data/word_embeddings/')]



MODEL_DIR = './models/'
CORPORA_DIR = './data/corpora/'

class ModelBuilder(cmd.Cmd):

    default_prompt = '(no model)$ '

    prompt = default_prompt

    def do_select(self, line):
        '''
        Will show a list of possible models to load (if any exist).
        If a model name is initially provided, the model will be loaded instead.

        if a model is selected, model versions will be shown to select from.
        '''

        if self.has_model_selected():
            m = ModelManager(self.prompt[:-1])
            selection = self.select_in_dir(m.folders['model_versions'], type='files')

            if not selection:
                return False

            for root, dirs, files in os.walk(m.folders['current_version']):
                for f in files:
                    logging.debug('removing %s from selection'%f)
                    os.unlink(os.path.join(root, f))

            logging.debug('copying %s to %s'%(selection, m.folders['current_version']))
            shutil.copy(m.folders['model_versions']+selection, m.folders['current_version']+selection)

            return False

        # list existing models
        if line == '':
            selection = self.select_in_dir(MODEL_DIR)
        else:
            selection = line

        if selection == False:
            print 'selection failed'
            return False

        model_loc = MODEL_DIR+selection

        if not os.path.exists(model_loc):
            print 'could not find', model_loc
            return False

        if self.has_model_selected():
            logging.debug('dropping already selected model')

        print 'selecting', model_loc

        m = ModelManager(selection)
        self.prompt = selection + '$'

    def do_create(self, line):
        '''
        Will create a new model directory structure for the specified name
        '''

        new_dir = MODEL_DIR+line

        if os.path.exists(new_dir):
            print 'Model already exists!'
            return False
        logging.debug('creating folder structure')
        manager = ModelManager(line)
        logging.debug('model created')
        self.prompt = line + '$'

    def do_state(self, line):
        '''
        Will link the specified model configuration (see state.py for an overview) to the selected model.
        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        if line == '':
            print 'Please provide a model configuration name as input (see state.py for an overview).'
            return False

        if os.path.exists(MODEL_DIR+self.prompt+'/state.txt'):
            print 'Model configuration already specified!'
            user_input = raw_input('overwrite? (y/n):')
            if not user_input.strip().startswith('y'):
                print 'aborting'
                return False

        manager = ModelManager(self.prompt[:-1])
        manager.select_state(line)


    def do_load_data(self, line):
        '''
        Loads textual data and converts it into a format that can be used by the model.

        optional arguments are:
        -train_set_size <percentage as float>
        -valid_set_size <percentage as float>

        example:
        -train_set_size 0.85 -valid_set_size 0.1

        set sizes: train 85%, valid 10%, and test 5%
        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        parser = argparse.ArgumentParser()
        parser.add_argument('-train_set_size', type=float,
                            help='The size of the training set given by a floating point number between 0 and 1.',
                            default=0.85)
        parser.add_argument('-valid_set_size', type=float,
                            help='The size of the validation set given by a floating point number between 0 and 1.',
                            default=0.10)

        input = parser.parse_args(line.split())

        corpus_selection = self.select_in_dir(CORPORA_DIR, type='files')

        if corpus_selection == False:
            print 'selection failed'
            return False


        m = ModelManager(self.prompt[:-1])

        logging.debug('copying textual data...')
        shutil.copy(CORPORA_DIR+corpus_selection, m.folders['data'])

        logging.debug('Train set size: %.3f, Validation set size: %.3f, the remaining data is used for the test set'%(input.train_set_size, input.valid_set_size))

        corpora_processing.convert_to_binarized_data(m, m.folders['data']+corpus_selection, m.folders['binarized'], train_set_size=input.train_set_size, valid_set_size=input.valid_set_size)

    def do_train_word2vec(self, line):
        '''
        Given the training data of the currently selected model,
        word embeddings will be trained using the gensim word2vec library.

        One can specify the feature length (default is 300):

            train_word2vec <feature_length>
            train_word2vec 300


        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        try: feature_length = int(line)
        except: feature_length = 300

        word_embedding_tools.train_embeddings(ModelManager(self.prompt[:-1]), feature_length)

    def do_load_pretrained_word_embeddings(self, line):
        '''
        Will load the word embeddings from a pre-trained model that fit to the vocabulary
        of the currently selected model. The resulting embeddings file will be stored
        in the model's folder and can be selected with select_word_embeddings

        If executed with: --fix_pretrained , pre-trained word embeddings will not
        be tuned during training.
        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        fix_pretrained = False
        if line != '':
            if line == '--fix_pretrained':
                fix_pretrained = True
            else:
                print 'Did not understand input: ', line
                return False


        m = ModelManager(self.prompt[:-1])

        selection = self.select_in_dir(m.folders['pre_trained_word_embeddings'], type='files')

        if selection:
            file_path = m.folders['pre_trained_word_embeddings'] + selection
        else:
            print 'selection failed'
            return False

        word_embedding_tools.load_pretrained_embeddings(m, file_path, fix_pretrained)


    def do_select_word_embeddings(self, line):
        '''
        Will give an option over the available word embeddings.
        Once a selection was made, the state file will be updated to reflect that selection.
        '''

        if not self.has_model_selected():
            print 'please select or create a model'
            return False
        m = ModelManager(self.prompt[:-1])

        selection = self.select_in_dir(m.folders['word_embeddings'], type='files')

        if selection:
            file_path = m.folders['word_embeddings'] + selection
        else:
            print 'selection failed'
            return False

        m.set_state_variable('pretrained_word_embeddings_file', file_path)
        m.set_state_variable('initialize_from_pretrained_word_embeddings', True)

    def do_train(self, line):
        '''
        Will start the training using the configuration stored in state.txt
        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        parser = argparse.ArgumentParser()
        parser.add_argument('-gui', action='store_true',
                            help='The size of the training set given by a floating point number between 0 and 1.',
                            default=False)
        args = parser.parse_args(line)

        m = ModelManager(self.prompt[:-1])
        state = m.load_current_state(add_hidden=True)

        train.train2(m, state, None)


        print

    def do_continue(self, line):
        '''
        Will continue the training of the currently selected model
        '''

        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        m = ModelManager(self.prompt[:-1])
        state = m.load_current_state(add_hidden=True)
        model = m.load_currently_selected_model()
        train.train2(m, state, model)

    def do_build_database(self, input):
        '''

        Will build a database from scratch storing binarized dialogues for quick access,
        as well as an indexing structures.
        '''

        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        m = ModelManager(self.prompt[:-1])
        build_database_from_scratch(m)

    def do_encode_corpus(self, input):
        '''
        Will encode the currently selected corpus and save the resulting embeddings to disk
        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        m = ModelManager(self.prompt[:-1])
        save_embeddings_to_file(m)
        check_embeddings_consistency(m)

    def do_build_lshf_model(self, input):
        '''
        Will train a LSH-Forest on a part of the corpus

        You can specify the percentage of the corpus to be used by

        build_lshf_model <percentage>

        build_lshf_model 0.1

        would use 10% of the corpus to train the model.
        '''
        if not self.has_model_selected():
            print 'please select or create a model'
            return False

        m = ModelManager(self.prompt[:-1])

        if input == '':
            percentage = 1.0
        else:
            percentage = float(input)

        lsh_forest.train_lsh_forest(m, corpus_percentage=percentage)
        lsh_forest.save_linked_utterance_embeddings(m)

    def do_download(self, input):
        '''
        Shows a selection of possible downloads.
        After selection, downloads the corresponding file.
        '''

        for idx, (download_name, link, file_name, save_dir) in enumerate(DOWNLOADS):
            print '(%i) %s'%(idx, download_name)

        user_input = raw_input('select number:')
        try:
            selection = int(user_input)
        except:
            print 'invalid input:', user_input
            return

        download_name, link, file_name, save_dir = DOWNLOADS[selection]

        logging.debug('Attempting to download %s'%link)

        response = requests.get(link, stream=True)

        with open(save_dir+file_name, "wb") as f:
            pbar = tqdm(unit="B", total=int(response.headers['Content-Length']))
            for chunk in response.iter_content(chunk_size=256):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)



    def has_model_selected(self):
        return self.prompt != self.default_prompt


    def select_in_dir(self, folder_path, type='folder'):
        '''
            Request the user to select a folder or file within a specified directory

        :param folder_path: the directory in which files or folders shall be selected
        :param type: either 'folder' or 'files'
        :return:
        '''
        (dirpath, dirnames, filenames) = os.walk(
            folder_path).next()

        if type is 'folder':
            select_from = dirnames
        elif type is 'files':
            select_from = filenames
        else:
            raise Exception('wrong type description, either \'folder\' or \'files\'')

        for idx, name in enumerate(select_from):
            print '(%i) %s' % (idx, name)

        if len(select_from) == 0:
            return None

        user_input = raw_input('select number:')
        try:
            return select_from[int(user_input)]
        except:
            print 'invalid input:', user_input
            return None




if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    ModelBuilder().cmdloop()