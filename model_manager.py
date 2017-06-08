from hred_vhred import state
import utils
import logging
import os
import cPickle
MODEL_DIR = './models/'

FOLDER_STRUCTURE='''
model_versions
 current_version
nearest_neighbor
data
 binarized
 evaluation
 word_embeddings
 nlp_models
'''

RELEVANT_FILES = [('state', 'state.txt', ''),
                  ('dict', 'dict.pkl', 'data/binarized/'),
                  ('train', 'train.dialogues.pkl', 'data/binarized/'),
                    ('valid', 'valid.dialogues.pkl', 'data/binarized/'),
                  ('test', 'test.dialogues.pkl', 'data/binarized/'),
                  ('word_embeddings', 'word2vec.bin', 'data/word_embeddings/'),
                  ('frequencies', 'frequencies.pkl', 'data/nlp_models/')]

EXTERNAL_FOLDERS = [('pre_trained_word_embeddings', './data/word_embeddings/')]

class ModelManager():

    folders = {}
    files = {}

    def __init__(self, model_name):
        self.model_name = model_name
        self.init_folder_structure()
        self.index_important_files()


    def init_folder_structure(self):

        root_dir = MODEL_DIR+self.model_name
        utils.create_folder_if_not_existing(root_dir)

        folder_list = [None for _ in xrange(10)]
        folder_list[0] = root_dir

        for line in FOLDER_STRUCTURE.splitlines():
            if line == '':
                continue

            folder_name = line.strip()
            tabs = line.count(' ')

            folder_list[tabs+1] = folder_name

            new_dir = '/'.join(folder_list[:tabs+2])

            utils.create_folder_if_not_existing(new_dir)
            self.folders[folder_name] = new_dir+'/'

        for folder_reference, folder_path in EXTERNAL_FOLDERS:
            self.folders[folder_reference] = folder_path


    def index_important_files(self):
        DIR = MODEL_DIR+self.model_name+'/'
        for descriptor, file_sub_string, file_dir in RELEVANT_FILES:
            target_dir = DIR+file_dir

            (dirpath, dirnames, filenames) = os.walk(target_dir).next()

            for filename in filenames:
                if file_sub_string in filename:
                    self.files[descriptor] = target_dir+filename
                    break



    def select_state(self, state_name):

        state_name= 'state.%s()'%state_name
        try:
            state = eval(state_name)
        except:
            print 'invalid state description:', state_name
            return

        self.save_state(state)



    def save_state(self, state):
        self.files['state'] = MODEL_DIR + self.model_name + '/state.txt'

        # load state file if existing or create one
        state_file = open(self.files['state'], 'wb')

        # delete content of file
        state_file.seek(0)
        state_file.truncate()

        logging.debug('saving current state to %s' % self.files['state'])

        for key, value in state.iteritems():
            value_executable = value
            if type(value) is str:
                value_executable = '\'' + value + '\''
            state_file.write('state[\'' + key + '\'] = ' + str(value_executable) + os.linesep)

        state_file.close()

    def load_current_state(self, add_hidden=False):
        if not 'state' in self.files:
            print 'could not find configuration file (state.txt)'

        state = {}

        for line in open(self.files['state']):
            exec(line)

        if add_hidden:
            state['train_dialogues'] = self.files['train']
            state['test_dialogues'] = self.files['test']
            state['valid_dialogues'] = self.files['valid']
            state['dictionary'] = self.files['dict']

        return state

    def set_state_variable(self, variable_name, variable_value):
        state = self.load_current_state()
        state[variable_name] = variable_value
        self.save_state(state)



    def load_vocabulary(self):

        with open(self.files['dict'], 'rb') as f:
            vocab = cPickle.load(f)

        return vocab

    def load_train_data(self):
        with open(self.files['train'], 'rb') as f:
            train = cPickle.load(f)

        return train

if __name__ == '__main__':
    m = ModelManager('test')
    state = m.load_current_state()
    print

