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
from data.encoding_tools import encode
from hred_vhred.search import BeamSampler
from gc import collect
import numpy as np
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

def get_resonse_lshf_evaluator(model_manager, selection_method):
    ann = lsh_forest.load_lshf(model_manager)
    utt_embs = lsh_forest.load_utterance_embeddings(model_manager)

    def evaluate(instance):
        distances, labels, embeddings = ann.kneighbors(instance['context_emb'], 120)
        labels = [(label[0], label[1] + 1) for label in labels]
        search_context = {'distances': distances,
                          'labels': labels,
                          'candidate_dialogue_embeddings': embeddings,
                          'utterance_embeddings': utt_embs,
                          'original_utterance_embedding': instance['question_utterance_emb'],
                          'original_dialogue_embedding': instance['context_emb']}
        scored = selection_method(search_context)
        #scored = answer_relevance(search_context)
        #scored = context_relevance(search_context)
        #scored = context_and_answer_relevance(search_context)

        scored = sorted(scored, key=lambda tpl: tpl[0])
        return scored, utt_embs

    return evaluate
def get_response_generator(encoder, return_embedding = True):

    sampler = BeamSampler(encoder)

    def answer_to(context):
        if isinstance(context, basestring):
            context = context.strip().split()
            if context[0] != '</s>':
                context = [encoder.end_sym_utterance] + context

            if context[-1] != '</s>':
                context = context +[encoder.end_sym_utterance]

        samples, costs = sampler.sample([context], n_samples=5, n_turns=1, ignore_unk=False)

        answer = samples[0][0]
        cost = costs[0][0]

        if return_embedding:
            embs = encode(answer, encoder)
            utt_emb = embs[1][-1][-1]
            return cost, answer, utt_emb

        return cost, answer

    return answer_to

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
def evaluate_generative(model_manager):
    rand_iter = random_response_generator(model_manager)
    encoder = model_manager.load_currently_selected_model()

    answer_model = get_response_generator(encoder)

    #translator = data_access.get_label_translator(model_manager)

    #evaluator = get_response_evaluator(model_manager.load_currently_selected_model())
    rankings = []
    start_time = time()
    for instance in evaluation_sample_iterator(model_manager):

        random_responses = [rand_iter.next() for x in xrange(9)]

        context = instance['context']

        cost, answer, pred_utt_emb = answer_model(context)

        candidates = [(cosine(pred_utt_emb, instance['answer_utterance_emb']), True)]
        for random_resp, rand_utt_emb in random_responses:
            cost = cosine(pred_utt_emb, rand_utt_emb)
            candidates.append((cost, False))

        candidates = sorted(candidates, key=lambda pair: pair[0])

        rank = [idx for idx, cand in enumerate(candidates) if candidates[idx][1]][0]
        rankings.append(rank)

        rATk = calculate_recall_at_k(rankings, 10)
        result_str = ' | '.join(['R@%i %.3f%%' % (k + 1, percentage * 100) for k, percentage in rATk.iteritems()])

        print_progress_bar(instance['progress'], instance['conversations'], additional_text=result_str,
                           start_time=start_time)


def evaluate(model_manager):

    rand_iter = random_response_generator(model_manager)
    #    encoder = model_manager.load_currently_selected_model()

    translator = data_access.get_label_translator(model_manager)

    evaluator = get_response_evaluator(model_manager.load_currently_selected_model())
    rankings = []
    start_time = time()
    progress = 0

    result_arr = FileArray('./results/decoder_results_%s.bin'%model_manager.model_name, shape=(20000, 1), dtype='i4')
    result_arr.open()


    for instance in evaluation_sample_iterator(model_manager):


        prev_result = result_arr.read(progress)

        if prev_result >= 1:
            progress += 1
            rankings.append(prev_result[0]-1)
            continue

        progress += 1
        random_responses = [rand_iter.next()[0] for x in xrange(9)]

        context = instance['context']
        candidates = [(evaluator(instance['answer'], context), True)]

        '''

        test = encode(context, encoder)
        test2 = encode(context + ' </s> ' + instance['answer'], encoder)


        print 'context emb', sum(test[0][0][0]), sum(instance['context_emb'])
        print 'question emb', sum(test[1][0][0]), sum(instance['question_utterance_emb'])
        print 'answer emb', sum(test2[1][-1][0]), sum(instance['answer_utterance_emb'])
        print 'answer context emb', sum(test2[0][-1][0]), sum(instance['answer_context_emb'])
        '''
        for random_resp in random_responses:
            cost = evaluator(random_resp, context)
            candidates.append((cost, False))

        candidates = sorted(candidates, key=lambda pair: pair[0])

        rank = [idx for idx, cand in enumerate(candidates) if candidates[idx][1]][0]
        rankings.append(rank)


        result_arr.write(progress-1, np.array([rank+1], dtype='i4'))

        rATk = calculate_recall_at_k(rankings, 10)
        result_str = ' | '.join(['R@%i %.3f%%' % (k + 1, percentage * 100) for k, percentage in rATk.iteritems()])

        print_progress_bar(instance['progress'], instance['conversations'], additional_text=result_str, start_time=start_time)

        if progress % 300 == 0:
            print 'gc collect'
            collect()

    result_arr.close()

def evaluate_lshf(model_manager, selection_method):
    rand_iter = random_response_generator(model_manager)

    evaluator = get_resonse_lshf_evaluator(model_manager, selection_method)

    rankings = []
    start_time = time()

    #translator = data_access.get_label_translator(model_manager)
    progress = 0

    result_arr = FileArray('./results/%s_%s.bin'%(selection_method.__name__,model_manager.model_name), shape=(20000, 1), dtype='i4')
    result_arr.open()

    for instance in evaluation_sample_iterator(model_manager):
        prev_result = result_arr.read(progress)

        if prev_result >= 1:
            progress += 1
            rankings.append(prev_result[0]-1)
            continue

        progress += 1

        random_responses = [rand_iter.next() for x in xrange(9)]


        candidates, utt_embs_set = evaluator(instance)
        '''
        print instance['context'][-160:]
        print
        for score, label in candidates[:10]:
            utt_emb_candidate = utt_embs_set[label[0]][label[1]]
            print score, cosine_similarity(utt_emb_candidate, instance['answer_utterance_emb']), translator(label)[0:100]
        '''
        label = candidates[0][1]
        predicted_utt_emb = utt_embs_set[label[0]][label[1]]

        '''
        print instance['context']
        print
        print instance['answer']
        print translator(label)
        print '*'*10
        '''
        candidates = [(cosine_similarity(instance['answer_utterance_emb'], predicted_utt_emb), True)]

        '''
        print
        print candidates[0][0], instance['answer'][0:100]
        print
        '''

        for rand_answer, random_utt_emb in random_responses:
            cost = cosine_similarity(random_utt_emb, predicted_utt_emb)
            #print cost, rand_answer
            candidates.append((cost, False))


        candidates = sorted(candidates, key=lambda pair: pair[0])

        rank = [idx for idx, cand in enumerate(candidates) if candidates[idx][1]][0]
        result_arr.write(progress-1, np.array([rank+1], dtype='i4'))
        #print rank
        #print '*'*50
        rankings.append(rank)

        rATk = calculate_recall_at_k(rankings, 10)
        result_str = ' | '.join(['R@%i %.3f%%' % (k + 1, percentage * 100) for k, percentage in rATk.iteritems()])

        print_progress_bar(instance['progress'], instance['conversations'], additional_text=result_str, start_time=start_time)

    result_arr.close()
#CAR 5%
#5.04% R@1 26.522% | R@2 41.583% | R@3 52.072% | R@4 60.903% | R@5 69.620% | R@6 76.993% | R@7 83.510% | R@8 89.854% | R@9 95.456% remaining time: 1:30:22Traceback (most recent call last):
