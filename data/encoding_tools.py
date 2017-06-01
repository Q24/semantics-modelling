import numpy as np

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