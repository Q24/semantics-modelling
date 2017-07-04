import data.data_access as data_access
from data.data_file import FileArray
from random import Random
from logging import debug
from hred_vhred import search
from time import time
from utils import print_progress_bar
from hred_vhred import search
from ann import lsh_forest
from ann.candidate_selection import *

def evaluation_sample_iterator(model_manager, amount = 30000, seed = 10):
    rand = Random(seed)

    database = data_access.get_database(model_manager)

    test_ids = database[data_access.TEST_IDS_SET_NAME][:]
    rand.shuffle(test_ids)
    test_ids = test_ids[0:min(len(test_ids), amount)]

    debug('yielding evaluation samples from %i conversations'%len(test_ids))

    coords = [database[data_access.EMBEDDINGS_COORDINATES_SET_NAME][d_idx] for d_idx in test_ids]

    utt_embs = FileArray(model_manager.files['utterance_embeddings'])
    dia_embs = FileArray(model_manager.files['dialogue_embeddings'])
    utt_embs.open()
    dia_embs.open()

    label_to_text = data_access.get_label_translator(model_manager)

    #context-textual response-textual, response-emb context-emb, answer-emb
    progress = 0
    for d_idx, (global_idx, conv_length) in zip(test_ids, coords):
        progress += 1
        context = label_to_text((d_idx, 0))

        relevant_utt_embs = utt_embs.read_chunk(global_idx, conv_length)
        relevant_dia_embs = dia_embs.read_chunk(global_idx, conv_length)

        for idx, conv_turn in enumerate(xrange(0, conv_length-1)):
            instance = {}
            instance['question'] = label_to_text((d_idx, conv_turn))
            instance['question_utterance_emb'] = relevant_utt_embs[idx]
            instance['context'] = context
            instance['context_emb'] = relevant_dia_embs[idx]
            instance['answer'] = label_to_text((d_idx, conv_turn+1))
            instance['answer_utterance_emb'] = relevant_utt_embs[idx+1]
            instance['answer_context_emb'] = relevant_dia_embs[idx+1]
            instance['progress'] = progress
            instance['conversations'] = len(test_ids)

            yield instance

            context = context + ' </s> ' + instance['answer']

    utt_embs.close()
    dia_embs.close()


def random_response_generator(model_manager, seed = 10):
    rand = Random(seed)

    database = data_access.get_database(model_manager)

    test_ids = database[data_access.TEST_IDS_SET_NAME][:]

    debug('yielding random responses from %i conversations' % len(test_ids))

    coords = [database[data_access.EMBEDDINGS_COORDINATES_SET_NAME][d_idx] for d_idx in test_ids]

    utt_embs = FileArray(model_manager.files['utterance_embeddings'])
    utt_embs.open()

    label_to_text = data_access.get_label_translator(model_manager)

    while 1:

        idx = rand.randint(0, len(test_ids)-1)
        d_idx = test_ids[idx]
        global_idx, conv_length = coords[idx]

        conv_turn = rand.randint(0, conv_length-1)
        yield label_to_text((d_idx, conv_turn)), utt_embs.read(global_idx+conv_turn)

def get_resonse_lshf_evaluator(model_manager):
    ann = lsh_forest.load_lshf(model_manager)
    utt_embs = lsh_forest.load_utterance_embeddings(model_manager)

    def evaluate(instance):
        distances, labels, embeddings = ann.kneighbors(instance['context_emb'], 10)
        labels = [(label[0], label[1] + 1) for label in labels]
        search_context = {'distances': distances,
                          'labels': labels,
                          'candidate_dialogue_embeddings': embeddings,
                          'utterance_embeddings': utt_embs,
                          'original_utterance_embedding': instance['question_utterance_emb'],
                          'original_dialogue_embedding': instance['context_emb']}
        #scored = answer_relevance(search_context)
        scored = context_relevance(search_context)

        scored = sorted(scored, key=lambda tpl: tpl[0])
        label = scored[0][1]
        return utt_embs[label[0]][label[1]], label

    return evaluate

def get_response_evaluator(encoder):

    evaluator = search.CostCalculator(encoder)

    def score_response(response, context):
        if isinstance(context, basestring):
            context = context.strip().split()
            if context[0] != '</s>':
                context = [encoder.end_sym_utterance] + context

            if context[-1] != '</s>':
                context = context +[encoder.end_sym_utterance]



        evaluator.set_response(response)


        samples, costs = evaluator.sample([context], n_samples=1, n_turns=1, ignore_unk=False)

        return costs[0][0]

    return score_response

def calculate_recall_at_k(rankings, num_candidates):

    measurements = {n:0 for n in xrange(num_candidates-1)}

    for ranking in rankings:

        for k, v in measurements.iteritems():
            if k >= ranking:
                measurements[k] += 1

    for k, v in measurements.iteritems():

        measurements[k] /= float(len(rankings))

    return measurements

def evaluate(model_manager):

    rand_iter = random_response_generator(model_manager)

    evaluator = get_response_evaluator(model_manager.load_currently_selected_model())
    rankings = []
    start_time = time()
    for instance in evaluation_sample_iterator(model_manager):

        random_responses = [rand_iter.next()[0] for x in xrange(9)]

        context = instance['context']


        candidates = [(evaluator(instance['answer'], context), True)]
        for random_resp in random_responses:
            cost = evaluator(random_resp, context)
            candidates.append((cost, False))

        candidates = sorted(candidates, key=lambda pair: pair[0])

        rank = [idx for idx, cand in enumerate(candidates) if candidates[idx][1]][0]
        rankings.append(rank)

        rATk = calculate_recall_at_k(rankings, 10)
        result_str = ' | '.join(['R@%i %.3f%%' % (k + 1, percentage * 100) for k, percentage in rATk.iteritems()])

        print_progress_bar(instance['progress'], instance['conversations'], additional_text=result_str, start_time=start_time)




def evaluate_lshf(model_manager):
    rand_iter = random_response_generator(model_manager)

    evaluator = get_resonse_lshf_evaluator(model_manager)

    rankings = []
    start_time = time()

    translator = data_access.get_label_translator(model_manager)
    for instance in evaluation_sample_iterator(model_manager):

        random_responses = [rand_iter.next()[1] for x in xrange(9)]

        predicted_utt_emb, label = evaluator(instance)
        '''
        print instance['context']
        print
        print instance['answer']
        print translator(label)
        print '*'*10
        '''
        candidates = [(cosine_similarity(instance['answer_utterance_emb'], predicted_utt_emb), True)]
        for random_utt_emb in random_responses:
            cost = cosine_similarity(random_utt_emb, predicted_utt_emb)
            candidates.append((cost, False))

        candidates = sorted(candidates, key=lambda pair: pair[0])

        rank = [idx for idx, cand in enumerate(candidates) if candidates[idx][1]][0]
        rankings.append(rank)

        rATk = calculate_recall_at_k(rankings, 10)
        result_str = ' | '.join(['R@%i %.3f%%' % (k + 1, percentage * 100) for k, percentage in rATk.iteritems()])

        print_progress_bar(instance['progress'], instance['conversations'], additional_text=result_str, start_time=start_time)



#Ubuntu vhred
#R@1 18.098% | R@2 31.862% | R@3 43.254% | R@4 54.293% | R@5 63.805% | R@6 72.826% | R@7 80.594% | R@8 88.035% | R@9 94.167% remaining time: 14:56:29Traceback (most recent call last):
