import numpy as np
import logging
from data_file import FileArray
import numpy as np
from utils import batch_iterator, print_progress_bar
from data_access import *
from gc import collect
from data_file import FileArray
from random import sample

def encode(dialogue, model, as_text=True, all_dialogue_embeddings=True, test_values_enabled = False, only_relevant_embeddings = True):
    '''
    will return dialogue and utterance embeddings of a textual dialogue (can also be word-indices as_text=False)
    given a model.
    '''


    #encoding_function = model.build_encoder_function2()

    if as_text:
        dialogue_splitted = dialogue.strip().split()
        dialogue_indices = model.words_to_indices(dialogue_splitted)
    else:
        dialogue_indices = dialogue

    if dialogue_indices[0] is not model.eos_sym:
        dialogue_indices = np.insert(dialogue_indices, 0, model.eos_sym)

    if dialogue_indices[-1] is not model.eos_sym:
        dialogue_indices = np.append(dialogue_indices, model.eos_sym)

    #dialogue_indices_reversed = model.reverse_utterances_indices(dialogue_indices)
    dialogue_indices_reversed = dialogue_indices


    max_length = len(dialogue_indices)

    updated_context = np.array(dialogue_indices, ndmin=2, dtype="int32").T
    updated_context_reversed = np.array(dialogue_indices_reversed, ndmin=2, dtype="int32").T

    if test_values_enabled:
        encoding_function = model.build_encoder_function(test_values={'x_data':updated_context})
    else:
        encoding_function = model.build_encoder_function()

    encoder_states = encoding_function(updated_context, updated_context_reversed, max_length)

    if not only_relevant_embeddings:
        return encoder_states

    if all_dialogue_embeddings:
        utterance_embeddings = retrieve_all_relevant_embeddings(encoder_states[0], dialogue_indices, model)
        dialogue_embedding = retrieve_all_relevant_embeddings(encoder_states[1], dialogue_indices, model)
    else:
        utterance_embeddings = retrieve_all_relevant_embeddings(encoder_states[0], dialogue_indices, model)
        dialogue_embedding = retrieve_last_relevant_embedding(encoder_states[1], dialogue_indices, model)[-1]
    return dialogue_embedding, utterance_embeddings

def encode_batch_to_embeddings(model, indices_batch):
    # data to encode, shape=(longest_dialogue, dialogues_in_batch)
    max_length = max([len(indices) for indices in indices_batch])
    batch_size = len(indices_batch)
    x_data = np.zeros((max_length, batch_size), dtype='int32')
    x_data_reversed = np.zeros((max_length, batch_size), dtype='int32')

    for idx, indices in enumerate(indices_batch):
        x_data[:len(indices), idx] = np.asarray(indices)

        #reversed = model.reverse_utterances_indices(indices)
        reversed = indices[:]


        assert len(reversed) == len(indices)

        x_data_reversed[:len(reversed), idx] = np.asarray(reversed)



    enc_func = model.build_encoder_function()

    # encode batch (n=batch_size)
    res = enc_func(x_data, x_data_reversed, max_length)
    h_n = res[0]
    hs_n = res[1]

    encodings = []

    for batch_idx in xrange(batch_size):

        indices = indices_batch[batch_idx]
        h = h_n[:max_length, batch_idx]

        hs = hs_n[:max_length, batch_idx]
        #hs = hs_comp_n[:max_length, batch_idx]
        #hs = hd_n[:max_length, batch_idx]

        utterance_embeddings = retrieve_all_relevant_embeddings(h, indices, model)
        dialogue_embeddings = retrieve_all_relevant_embeddings(hs, indices, model)

        # dialogue_idx, text, indices, utterance_embeddings, dialogue_embeddings

        encodings.append((utterance_embeddings, dialogue_embeddings))

    return encodings

def retrieve_last_relevant_embedding(dialogue_encoder_states, dialogue_indices, model):

    final_token_idx = len(dialogue_indices)-1
    for idx, token_idx in enumerate(dialogue_indices):

        if token_idx == model.eod_sym:
            final_token_idx = idx
            break

    return dialogue_encoder_states[final_token_idx]

def retrieve_all_relevant_embeddings(utterance_encoder_states, dialogue_indices, model):
    embeddings = []
    for idx, token_idx in enumerate(dialogue_indices):
        if token_idx == model.eos_sym and idx != 0:
            embeddings.append(utterance_encoder_states[idx])

    return embeddings

def save_embeddings_to_file(model_manager, encoding_batch_size = 20):

    state = model_manager.load_current_state()

    utt_features = state['qdim_encoder']
    dia_features = state['sdim']

    encoder = model_manager.load_currently_selected_model()

    database = get_database(model_manager)

    dset = database[BINARIZED_SET_NAME]

    logging.debug('loading binarized dialogues into memory and storing index')
    binarized = [(d_idx, indices) for d_idx, indices in enumerate(dset)]


    logging.debug('...creating and storing mapping from (dialogue id and utterance id) to (embedding storage location)')

    del_if_exists(database, EMBEDDINGS_COORDINATES_SET_NAME)
    coords_set = database.create_dataset(EMBEDDINGS_COORDINATES_SET_NAME,  (len(binarized),2), dtype='i4')

    enlargened_idx = 0

    progress = 0
    start_time = time()

    for d_idx, indices in binarized:
        progress += 1

        conv_length = num_turns(indices, encoder.eos_sym)
        coords = (enlargened_idx, conv_length)

        coords_set[d_idx] = coords

        enlargened_idx += conv_length
        if progress % 100 == 0:
            print_progress_bar(progress, len(binarized), additional_text='coords for %i dialogues created'%progress, start_time=start_time)

    num_embeddings = enlargened_idx

    logging.debug('...sorting based on dialouge length for quicker encoding')
    dialouge_ids, binarized = zip(*sorted(binarized, key=lambda tuple: len(tuple[1]), reverse=True))

    collect()

    logging.debug('creating files that hold embeddings')
    utt_file = FileArray(model_manager.folders['embeddings'] + 'utterance.embeddings.bin',
                         shape=(num_embeddings, utt_features), dtype='f4')

    dia_file = FileArray(model_manager.folders['embeddings'] + 'dialogue.embeddings.bin',
                         shape=(num_embeddings, dia_features), dtype='f4')

    utt_file.open()
    dia_file.open()

    progress = 0
    batches_to_process = len(binarized) / encoding_batch_size
    start_time = time()

    total_embeddings_encoded = 0

    for d_indices, batch in zip(batch_iterator(dialouge_ids, batch_size=encoding_batch_size),
                                batch_iterator(binarized, batch_size=encoding_batch_size,
                                               apply_on_element=lambda indices: np.append([encoder.eos_sym],
                                                                                          np.append(indices,
                                                                                                    encoder.eos_sym)))):
        progress += 1

        embeddings = encode_batch_to_embeddings(encoder, batch)

        coords = [coords_set[d_idx] for d_idx in d_indices]

        for embs, coord in zip(embeddings, coords):
            utt_embs = embs[0]
            dia_embs = embs[1]

            assert len(utt_embs) == coord[1]
            assert len(dia_embs) == coord[1]

            assert len(utt_embs[0]) == utt_features
            assert len(dia_embs[0]) == dia_features

            #for local_idx, global_idx in enumerate(xrange(coord[0], coord[0] + coord[1])):
            #    utt_file.write(global_idx, utt_embs[local_idx])
            #    dia_file.write(global_idx,  dia_embs[local_idx])
            #    total_embeddings_encoded += 1
            utt_file.write_chunk(coord[0], utt_embs)
            dia_file.write_chunk(coord[0], dia_embs)
            total_embeddings_encoded += coord[1]

        if progress % 1000 == 0:
            collect()

        # print '%i of %i batches encoded (%.3f%%)'%(progress, batches_to_process, 100*(float(progress)/float(batches_to_process)))
        print_progress_bar(progress, batches_to_process,
                           additional_text='%i batches of dialogues processed (total of %i dialogues) (total of %i embeddings) (%i conv length)' % (
                           progress, ((progress - 1) * encoding_batch_size) + len(batch), total_embeddings_encoded,
                           len(batch[0])), start_time=start_time)

    utt_file.close()
    dia_file.close()

def check_embeddings_consistency(model_manager, samples=1000, epsilon=1e-4):
    database = get_database(model_manager)
    state = model_manager.load_current_state()

    utt_features = state['qdim_encoder']
    dia_features = state['sdim']

    # emb data sets
    utt_file = FileArray(model_manager.folders['embeddings'] + 'utterance.embeddings.bin')
    dia_file = FileArray(model_manager.folders['embeddings'] + 'dialogue.embeddings.bin')



    # getting a random set of dialogue ids
    coord_set = database[EMBEDDINGS_COORDINATES_SET_NAME]
    rand_indices = sample(range(len(coord_set)), samples)
    rand_coords = [coord_set[idx] for idx in rand_indices]

    # get binarized dialogues
    binarized_set = database[BINARIZED_SET_NAME]
    binarized_dialogues = [binarized_set[idx] for idx in rand_indices]

    # init encoder
    encoder = model_manager.load_currently_selected_model()


    # check all stored embeddings against the expected embeddings
    progress = 0
    start_time = time()


    utt_file.open()
    dia_file.open()
    logging.debug('...checking all stored embeddings against the expected embeddings (EPSILON %s)'%str(epsilon))
    for d_idx, coord, indices in zip(*[rand_indices, rand_coords, binarized_dialogues]):
        progress += 1

        global_idx = coord[0]
        conv_length = coord[1]

        true_dialogue_embeddings, true_utterance_embeddings = encode(indices, encoder, as_text=False)

        #stored_utterance_embeddings = [utt_file.read(idx) for idx in xrange(global_idx, global_idx+conv_length)]
        #stored_dialogue_embeddings = [dia_file.read(idx) for idx in xrange(global_idx, global_idx+conv_length)]
        stored_utterance_embeddings = utt_file.read_chunk(global_idx, conv_length)
        stored_dialogue_embeddings = dia_file.read_chunk(global_idx, conv_length)

        assert len(true_utterance_embeddings) == len(stored_utterance_embeddings)
        assert len(true_dialogue_embeddings) == len(stored_dialogue_embeddings)
        assert len(true_utterance_embeddings) == len(true_dialogue_embeddings)
        assert len(true_utterance_embeddings) == conv_length

        for tr_u_emb, st_u_emb in zip(true_utterance_embeddings, stored_utterance_embeddings):
            #print sum(tr_u_emb[0]), sum(st_u_emb)
            assert abs(sum(tr_u_emb[0])-sum(st_u_emb)) < epsilon

        for tr_d_emb, st_d_emb in zip(true_dialogue_embeddings, stored_dialogue_embeddings):
            #print sum(tr_d_emb[0]), sum(st_d_emb)
            assert abs(sum(tr_d_emb[0])-sum(st_d_emb)) < epsilon


        print_progress_bar(progress, samples, additional_text="%i dialogue's embeddings checked"%progress, start_time=start_time)
    utt_file.close()
    dia_file.close()

def num_turns(indices, eos_sym=1):
    counts = sum(1 for value in indices if value == eos_sym)
    if indices[0] == eos_sym:
        counts -= 1

    if not indices[-1] == eos_sym:
        counts += 1
    return counts

def create_model_specific_encoding_hash(model):
    '''
    Generates a generic conversation from the model's vocabulary and encodes it.
    The returned hash is based on the generated utterance and dialogue embeddings.
    With this hash value, one can ensure that the loaded model generates the same embeddings
    as the trained one or, generally, check if two models are the same.
    '''

    generic_conversation = [1,2,3,4,5,1,2,3,4,5,1,1]

    embeddings = encode(generic_conversation, model, as_text=False)

    hash_value = 0.

    for dia_emb in embeddings[0]:
        hash_value += sum(dia_emb[0])

    for utt_emb in embeddings[1]:
        hash_value += sum(utt_emb[0])

    return hash_value

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    from model_manager import ModelManager

    m = ModelManager('ubuntu_vhred_vanilla')

    save_embeddings_to_file(m)

