# -*- coding: utf-8 -*-
#!/usr/bin/env python

from data_iterator import *
from state import *
from dialog_encdec import *
from utils import *

import time
import traceback
import sys
import argparse
import cPickle
import logging
import search
import pprint
import numpy
import collections
import signal
import math
import gc
import datetime

import os
import os.path

from os import listdir
from os.path import isfile, join


from data.encoding_tools import create_model_specific_encoding_hash
import matplotlib
matplotlib.use('Agg')
import pylab


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
logger = logging.getLogger(__name__)

### Unique RUN_ID for this execution
RUN_ID = str(time.time())

### Additional measures can be set here
measures = ["valid_perplexity", "train_progress", "train_cost", "train_misclass", "train_kl_divergence_cost", "train_posterior_mean_variance", "valid_cost", "valid_misclass", "valid_posterior_mean_variance", "valid_kl_divergence_cost", "valid_emi", "step"]


def init_timings():
    timings = {}
    for m in measures:
        timings[m] = []
    return timings

def save3(model, timings, model_manager):
    print 'Saving model...'

    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_')

    # this is to ensure that the validation will not started again after the model was loaded
    #timings['step'][0] += 1

    try:
        if len(timings["valid_perplexity"]) > 0:
            valid_cost = timings["valid_perplexity"][-1]
        else:
            valid_cost = float("inf")
    except:
        valid_cost = float("inf")

    name_post_fix = str(stamp) + str('%.3f_'%valid_cost)

    model.timings = timings
    model.timings['encoding_hash'] = create_model_specific_encoding_hash(model)
    logging.debug("model specific encoding value %.12f"% model.timings['encoding_hash'])
    cPickle.dump(model, open(model_manager.folders['model_versions']+name_post_fix+'model.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    #timings['step'][0] -=1

    gc.collect()

def save2(model, timings, commands):
    print "Saving the model..."

    model_config = commands['model_config']
    directory_path = model_config.model_dir + 'model_versions/'

    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_')

    # this is to ensure that the validation will not started again after the model was loaded
    timings['step'][0] += 1

    if len(timings["valid_perplexity"]) > 0:
        valid_cost = timings["valid_perplexity"][-1]
    else:
        valid_cost = float("inf")

    name_post_fix = str(stamp) + str('%.3f_'%valid_cost)

    if commands['save_as'] is 'pkl':
        model.timings = timings
        cPickle.dump(model, open(directory_path+name_post_fix+'model.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    else:
        model.save(directory_path+name_post_fix+'model.npz')
        cPickle.dump(model.state, open(directory_path+name_post_fix+'state.pkl','wb'))
        numpy.savez(directory_path+name_post_fix + 'timing.npz', **timings)

    timings['step'][0] -=1

    gc.collect()

def save(model, timings, post_fix = ''):
    print "Saving the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    #s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    model.save(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'model.npz')
    cPickle.dump(model.state, open(model.state['save_dir'] + '/' +  model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'state.pkl', 'wb'))
    numpy.savez(model.state['save_dir'] + '/' + model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'timing.npz', **timings)
    #signal.signal(signal.SIGINT, s)
    
    print "Model saved, took {}".format(time.time() - start)

def load(model, filename, parameter_strings_to_ignore):
    print "Loading the model..."

    # ignore keyboard interrupt while saving
    start = time.time()
    #s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename, parameter_strings_to_ignore)
    #signal.signal(signal.SIGINT, s)

    print "Model loaded, took {}".format(time.time() - start)



def train2(model_manager, state, model = None, random_seed = True):

    for k, v in state.iteritems():
        print k, '=', v

    if not model:
        model = DialogEncoderDecoder(state)
        step = 0
    else:
        step = model.timings['step']


    logger.debug("Compile trainer")
    if not state["use_nce"]:
        if ('add_latent_gaussian_per_utterance' in state) and (state["add_latent_gaussian_per_utterance"]):
            logger.debug("Training using variational lower bound on log-likelihood")
        else:
            logger.debug("Training using exact log-likelihood")

        train_batch = model.build_train_function()
    else:
        logger.debug("Training with noise contrastive estimation")
        train_batch = model.build_nce_function()

    eval_batch = model.build_eval_function()

    if model.add_latent_gaussian_per_utterance:
        eval_grads = model.build_eval_grads()

    random_sampler = search.RandomSampler(model)

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    train_data.start()

    # Start looping through the dataset
    patience = state['patience']
    start_time = time.time()

    train_cost = 0
    train_kl_divergence_cost = 0
    train_posterior_mean_variance = 0
    train_misclass = 0
    train_done = 0
    train_dialogues_done = 0.0

    prev_train_cost = 0
    prev_train_done = 0

    valid_rounds = 0

    ex_done = 0
    is_end_of_batch = True
    start_validation = False

    batch = None


    if random_seed:
        rng = numpy.random.RandomState()
    else:
        rng = model.rng

    timings = {'step':step}
    total_token_time = 0
    num_tokens_processed = 0

    while (step < state['loop_iters'] and
                       (time.time() - start_time) / 60. < state['time_stop'] and
                   patience >= 0):

        if 'save_at_iter' in state and step == state['save_at_iter']:
            save3(model, timings, model_manager)

        timings['step'] = step

        ### Sampling phase
        if step % 200 == 0:
            # First generate stochastic samples
            for param in model.params:
                print "%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5)

            samples, costs = random_sampler.sample([[]], n_samples=1, n_turns=3)
            print "Sampled : {}".format(samples[0])

        ### Training phase
        batch = train_data.next()

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got no batch to train with...")
            break

        logger.debug("[TRAIN] [STEP %d]- Got batch %d,%d" % (step + 1, batch['x'].shape[1], batch['max_length']))

        x_data = batch['x']

        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_reset = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']
        ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

        is_end_of_batch = False
        if numpy.sum(numpy.abs(x_reset)) < 1:
            # Print when we reach the end of an example (e.g. the end of a dialogue or a document)
            # Knowing when the training procedure reaches the end is useful for diagnosing training problems
            # print 'END-OF-BATCH EXAMPLE!'
            is_end_of_batch = True
        token_time = time.time()

        if state['use_nce']:
            y_neg = rng.choice(size=(10, max_length, x_data.shape[1]), a=model.idim, p=model.noise_probs).astype('int32')
            c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, y_neg, max_length, x_cost_mask, x_reset, ran_cost_utterance, ran_decoder_drop_mask)
        else:
            c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_cost_utterance, ran_decoder_drop_mask)


        total_token_time += (time.time()-token_time)
        num_tokens_processed +=(batch['x'].shape[1] * batch['max_length'])

        print '%.3f words/s' % (num_tokens_processed/total_token_time)


        # Print batch statistics
        print 'cost_sum', c
        print 'cost_mean', c / float(numpy.sum(x_cost_mask))
        print 'kl_divergence_cost_sum', kl_divergence_cost
        print 'kl_divergence_cost_mean', kl_divergence_cost / float(len(numpy.where(x_data == model.eos_sym)[0]))
        print 'posterior_mean_variance', posterior_mean_variance

        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            gc.collect()
            continue

        train_cost += c
        train_kl_divergence_cost += kl_divergence_cost
        train_posterior_mean_variance += posterior_mean_variance

        train_done += batch['num_preds']
        train_dialogues_done += batch['num_dialogues']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time

            # Keep track of training cost for the last 'train_freq' batches.
            current_train_cost = train_cost / train_done

            if prev_train_done >= 1 and abs(train_done - prev_train_done) > 0:
                current_train_cost = float(train_cost - prev_train_cost) / float(train_done - prev_train_done)

            if numpy.isinf(c) or numpy.isnan(c):
                current_train_cost = 0

            prev_train_cost = train_cost
            prev_train_done = train_done

            h, m, s = ConvertTimedelta(this_time - start_time)

            # We need to catch exceptions due to high numbers in exp
            try:
                print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f acc_word_perplexity = %.4f cur_cost = %.4f cur_word_perplexity = %.4f acc_mean_word_error = %.4f acc_mean_kl_divergence_cost = %.8f acc_mean_posterior_variance = %.8f" % (
                h, m, s, \
                state['time_stop'] - (time.time() - start_time) / 60., \
                step, \
                batch['x'].shape[1], \
                batch['max_length'], \
                float(train_cost / train_done), \
                math.exp(float(train_cost / train_done)), \
                current_train_cost, \
                math.exp(current_train_cost), \
                float(train_misclass) / float(train_done), \
                float(train_kl_divergence_cost / train_done), \
                float(train_posterior_mean_variance / train_dialogues_done))
            except:
                pass

            # timings['train_progress'].append(math.exp(float(train_cost/train_done)))
            #timings['train_progress'].append(math.exp(current_train_cost))

        ### Inspection phase
        # Evaluate gradient variance every 200 steps for GRU decoder
        if state['utterance_decoder_gating'].upper() == "GRU":
            if (step % 200 == 0) and (model.add_latent_gaussian_per_utterance):
                k_eval = 10

                softmax_costs = numpy.zeros((k_eval), dtype='float32')
                var_costs = numpy.zeros((k_eval), dtype='float32')
                gradients_wrt_softmax = numpy.zeros((k_eval, model.qdim_decoder, model.qdim_decoder), dtype='float32')
                for k in range(0, k_eval):
                    batch = add_random_variables_to_batch(model.state, model.rng, batch, None, False)
                    ran_cost_utterance = batch['ran_var_constutterance']
                    ran_decoder_drop_mask = batch['ran_decoder_drop_mask']
                    softmax_cost, var_cost, grads_wrt_softmax, grads_wrt_kl_divergence_cost = eval_grads(x_data,
                                                                                                         x_data_reversed,
                                                                                                         max_length,
                                                                                                         x_cost_mask,
                                                                                                         x_reset,
                                                                                                         ran_cost_utterance,
                                                                                                         ran_decoder_drop_mask)
                    softmax_costs[k] = softmax_cost
                    var_costs[k] = var_cost
                    gradients_wrt_softmax[k, :, :] = grads_wrt_softmax

                print 'mean softmax_costs', numpy.mean(softmax_costs)
                print 'std softmax_costs', numpy.std(softmax_costs)

                print 'mean var_costs', numpy.mean(var_costs)
                print 'std var_costs', numpy.std(var_costs)

                print 'mean gradients_wrt_softmax', numpy.mean(
                    numpy.abs(numpy.mean(gradients_wrt_softmax, axis=0))), numpy.mean(gradients_wrt_softmax, axis=0)
                print 'std gradients_wrt_softmax', numpy.mean(numpy.std(gradients_wrt_softmax, axis=0)), numpy.std(
                    gradients_wrt_softmax, axis=0)

                print 'std greater than mean', numpy.where(
                    numpy.std(gradients_wrt_softmax, axis=0) > numpy.abs(numpy.mean(gradients_wrt_softmax, axis=0)))[
                    0].shape[0]

                Wd_s_q = model.utterance_decoder.Wd_s_q.get_value()

                print 'Wd_s_q all', numpy.sum(numpy.abs(Wd_s_q)), numpy.mean(numpy.abs(Wd_s_q))
                print 'Wd_s_q latent', numpy.sum(numpy.abs(
                    Wd_s_q[(Wd_s_q.shape[0] - state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :])), numpy.mean(
                    numpy.abs(Wd_s_q[(Wd_s_q.shape[0] - state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :]))

                print 'Wd_s_q ratio', (numpy.sum(numpy.abs(
                    Wd_s_q[(Wd_s_q.shape[0] - state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :])) / numpy.sum(
                    numpy.abs(Wd_s_q)))

                if 'latent_gaussian_linear_dynamics' in state:
                    if state['latent_gaussian_linear_dynamics']:
                        prior_Wl_linear_dynamics = model.latent_utterance_variable_prior_encoder.Wl_linear_dynamics.get_value()
                        print 'prior_Wl_linear_dynamics', numpy.sum(numpy.abs(prior_Wl_linear_dynamics)), numpy.mean(
                            numpy.abs(prior_Wl_linear_dynamics)), numpy.std(numpy.abs(prior_Wl_linear_dynamics))

                        approx_posterior_Wl_linear_dynamics = model.latent_utterance_variable_approx_posterior_encoder.Wl_linear_dynamics.get_value()
                        print 'approx_posterior_Wl_linear_dynamics', numpy.sum(
                            numpy.abs(approx_posterior_Wl_linear_dynamics)), numpy.mean(
                            numpy.abs(approx_posterior_Wl_linear_dynamics)), numpy.std(
                            numpy.abs(approx_posterior_Wl_linear_dynamics))

                        # print 'grads_wrt_softmax', grads_wrt_softmax.shape, numpy.sum(numpy.abs(grads_wrt_softmax)), numpy.abs(grads_wrt_softmax[0:5,0:5])
                        # print 'grads_wrt_kl_divergence_cost', grads_wrt_kl_divergence_cost.shape, numpy.sum(numpy.abs(grads_wrt_kl_divergence_cost)), numpy.abs(grads_wrt_kl_divergence_cost[0:5,0:5])


        ### Evaluation phase
        if valid_data is not None and \
                                step % state['valid_freq'] == 0 and step > 1:
            start_validation = True

        # Only start validation loop once it's time to validate and once all previous batches have been reset
        if start_validation and is_end_of_batch:
            start_validation = False
            valid_data.start()
            valid_cost = 0
            valid_kl_divergence_cost = 0
            valid_posterior_mean_variance = 0

            valid_wordpreds_done = 0
            valid_dialogues_done = 0

            logger.debug("[VALIDATION START]")

            while True:
                batch = valid_data.next()

                # Validation finished
                if not batch:
                    break

                logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))

                x_data = batch['x']
                x_data_reversed = batch['x_reversed']
                max_length = batch['max_length']
                x_cost_mask = batch['x_mask']

                x_reset = batch['x_reset']
                ran_cost_utterance = batch['ran_var_constutterance']
                ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

                c, kl_term, c_list, kl_term_list, posterior_mean_variance = eval_batch(x_data, x_data_reversed, max_length,
                                                                                       x_cost_mask, x_reset,
                                                                                       ran_cost_utterance,
                                                                                       ran_decoder_drop_mask)

                # Rehape into matrix, where rows are validation samples and columns are tokens
                # Note that we use max_length-1 because we don't get a cost for the first token
                # (the first token is always assumed to be eos)
                c_list = c_list.reshape((batch['x'].shape[1], max_length - 1), order=(1, 0))
                c_list = numpy.sum(c_list, axis=1)

                words_in_dialogues = numpy.sum(x_cost_mask, axis=0)
                c_list = c_list / words_in_dialogues

                if numpy.isinf(c) or numpy.isnan(c):
                    continue

                valid_cost += c
                valid_kl_divergence_cost += kl_divergence_cost
                valid_posterior_mean_variance += posterior_mean_variance

                # Print batch statistics
                print 'valid_cost', valid_cost
                print 'valid_kl_divergence_cost sample', kl_divergence_cost
                print 'posterior_mean_variance', posterior_mean_variance

                valid_wordpreds_done += batch['num_preds']
                valid_dialogues_done += batch['num_dialogues']

            logger.debug("[VALIDATION END]")

            valid_cost /= valid_wordpreds_done
            valid_kl_divergence_cost /= valid_wordpreds_done
            valid_posterior_mean_variance /= valid_dialogues_done

            # We need to catch exceptions due to high numbers in exp
            try:
                print "** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, valid kldiv cost (per word) = %.8f, valid mean posterior variance (per word) = %.8f, patience = %d" % (
                float(valid_cost), float(math.exp(valid_cost)), float(valid_kl_divergence_cost),
                float(valid_posterior_mean_variance), patience)
            except:
                try:
                    print "** valid cost (NLL) = %.4f, patience = %d" % (float(valid_cost), patience)
                except:
                    pass

            timings["train_cost"].append(train_cost / train_done)
            timings["train_kl_divergence_cost"].append(train_kl_divergence_cost / train_done)
            timings["train_posterior_mean_variance"].append(train_posterior_mean_variance / train_dialogues_done)
            timings["valid_cost"].append(valid_cost)
            timings["valid_perplexity"].append(float(math.exp(valid_cost)))
            timings["valid_kl_divergence_cost"].append(valid_kl_divergence_cost)
            timings["valid_posterior_mean_variance"].append(valid_posterior_mean_variance)

            save3(model, timings, model_manager)

            # Reset train cost, train misclass and train done metrics
            train_cost = 0
            train_done = 0
            prev_train_cost = 0
            prev_train_done = 0

            # Count number of validation rounds done so far
            valid_rounds += 1

        step += 1

    logger.debug("All done, exiting...")


#--prototype vhred_test --save_every_valid_iteration
def train(args, state=None, commands=None):
    if commands:
        def shall_train():
            return commands['train']
        def shall_save():
            return commands['save']
        def shall_abort():
            return commands['abort']
        def saving_done():
            commands['save'] = False

    #logging.basicConfig(level = logging.DEBUG,
    #                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    if not state:
        state = eval(args.prototype)()

    timings = init_timings()

    auto_restarting = False
    if args.auto_restart:
        assert not args.save_every_valid_iteration
        assert len(args.resume) == 0

        directory = state['save_dir']
        if not directory[-1] == '/':
            directory = directory + '/' 

        auto_resume_postfix = state['prefix'] + '_auto_model.npz'

        if os.path.exists(directory):
            directory_files = [f for f in listdir(directory) if isfile(join(directory, f))]
            resume_filename = ''
            for f in directory_files:
                if len(f) > len(auto_resume_postfix):
                    if f[len(f) - len(auto_resume_postfix):len(f)] == auto_resume_postfix:
                        if len(resume_filename) > 0:
                            print 'ERROR: FOUND MULTIPLE MODELS IN DIRECTORY:', directory
                            assert False
                        else:
                            resume_filename = directory + f[0:len(f)-len('__auto_model.npz')]

            if len(resume_filename) > 0:
                logger.debug("Found model to automatically resume: %s" % resume_filename)
                auto_restarting = True
                # Setup training to automatically resume training with the model found
                args.resume = resume_filename
                # Disable training from reinitialization any parameters
                args.reinitialize_decoder_parameters = False
                args.reinitialize_latent_variable_parameters = False
            else:
                logger.debug("Could not find any model to automatically resume...")

    step = 0

    if args.resume != "":
        logger.debug("Resuming %s" % args.resume)
        
        state_file = args.resume + '_state.pkl'
        if commands:
            if commands['state_path']:
                state_file = commands['state_path']

        timings_file = args.resume + '_timing.npz'
        
        if os.path.isfile(state_file) and os.path.isfile(timings_file):
            logger.debug("Loading previous state")
            
            state = cPickle.load(open(state_file, 'r'))
            timings = dict(numpy.load(open(timings_file, 'r')))
            for x, y in timings.items():
                timings[x] = list(y)

            step = timings['step'][0]

            # Increment seed to make sure we get newly shuffled batches when training on large datasets
            state['seed'] = state['seed'] + 10

        else:
            raise Exception("Cannot resume, cannot find files!")



    logger.debug("State:\n{}".format(pprint.pformat(state)))
    logger.debug("Timings:\n{}".format(pprint.pformat(timings)))
 
    if args.force_train_all_wordemb == True:
        state['fix_pretrained_word_embeddings'] = False

    if state['test_values_enabled']:
        train_data, \
        valid_data, = get_train_iterator(state)
        train_data.start()
        state['batch_iterator'] = train_data


    if not commands:
        model = DialogEncoderDecoder(state)

    if commands:
        commands['timings'] = timings

        if commands['resume_path']:
            model = commands['resume_path'][0]
            timings = commands['resume_path'][1]

            for key, value in timings.iteritems():
                timings[key] = list(value)

            step = timings['step'][0]
        else:
            model = DialogEncoderDecoder(state)


    rng = model.rng 

    valid_rounds = 0
    save_model_on_first_valid = False

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            logger.debug("Loading previous model")

            parameter_strings_to_ignore = []
            if args.reinitialize_decoder_parameters:
                parameter_strings_to_ignore += ['Wd_']
                parameter_strings_to_ignore += ['bd_']

                save_model_on_first_valid = True
            if args.reinitialize_latent_variable_parameters:
                parameter_strings_to_ignore += ['latent_utterance_prior']
                parameter_strings_to_ignore += ['latent_utterance_approx_posterior']
                parameter_strings_to_ignore += ['kl_divergence_cost_weight']
                parameter_strings_to_ignore += ['latent_dcgm_encoder']

                save_model_on_first_valid = True

            load(model, filename, parameter_strings_to_ignore)
        else:
            raise Exception("Cannot resume, cannot find model file!")
        
        if 'run_id' not in model.state:
            print 'Backward compatibility not ensured! (need run_id in state)'

    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID

    logger.debug("Compile trainer")
    if not state["use_nce"]:
        if ('add_latent_gaussian_per_utterance' in state) and (state["add_latent_gaussian_per_utterance"]):
            logger.debug("Training using variational lower bound on log-likelihood")
        else:
            logger.debug("Training using exact log-likelihood")

        train_batch = model.build_train_function()
    else:
        logger.debug("Training with noise contrastive estimation")
        train_batch = model.build_nce_function()

    eval_batch = model.build_eval_function()

    if model.add_latent_gaussian_per_utterance:
        eval_grads = model.build_eval_grads()

    random_sampler = search.RandomSampler(model)
    beam_sampler = search.BeamSampler(model) 

    logger.debug("Load data")
    train_data, \
    valid_data, = get_train_iterator(state)
    train_data.start()



    # Start looping through the dataset
    patience = state['patience'] 
    start_time = time.time()
     
    train_cost = 0
    train_kl_divergence_cost = 0
    train_posterior_mean_variance = 0
    train_misclass = 0
    train_done = 0
    train_dialogues_done = 0.0

    prev_train_cost = 0
    prev_train_done = 0

    ex_done = 0
    is_end_of_batch = True
    start_validation = False

    batch = None

    import theano.tensor
    word = 'what'
    word_idx = model.words_to_indices([word])
    initial_sum = theano.tensor.sum(model.W_emb[word_idx]).eval()

    if 'fix_W_emb_steps' in state:
        model.W_emb_pretrained_mask.set_value(numpy.zeros(model.W_emb_pretrained_mask.shape.eval(), dtype='float32'))

    #for idx in xrange(10):
        #print theano.tensor.sum(model.W_emb[word_idx]).eval()

    total_token_time = 0
    num_tokens_processed = 0

    while (step < state['loop_iters'] and
            (time.time() - start_time)/60. < state['time_stop'] and
            patience >= 0):

        timings['step'] = [step]

        if 'save_at_first_iter' in state and step == 1:
            save2(model, timings, commands)

        #print 'init: ',initial_sum
        #print 'changed to: ',theano.tensor.sum(model.W_emb[word_idx]).eval()

        if 'fix_W_emb_steps' in state:
            if state['fix_W_emb_steps'] < step:
                model.W_emb_pretrained_mask.set_value(numpy.ones(model.W_emb_pretrained_mask.shape.eval(), dtype='float32'))

        if commands:

            commands['timings'] = timings

            if not shall_train():
                logging.debug('...training paused')
                wait_until(shall_train)

            if shall_save():
                logging.debug('...saving model (from command)')
                save2(model, timings, commands)
                saving_done()

            if shall_abort():
                break


        ### Sampling phase
        if step % 200 == 0:
            # First generate stochastic samples
            for param in model.params:
                print "%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5)

            samples, costs = random_sampler.sample([[]], n_samples=1, n_turns=3)
            print "Sampled : {}".format(samples[0])

            if commands:
                commands['output'] = samples[0]


        ### Training phase
        batch = train_data.next()

        # Train finished
        if not batch:
            # Restart training
            logger.debug("Got None...")
            break
        
        logger.debug("[TRAIN] [STEP %d]- Got batch %d,%d" % (step+1, batch['x'].shape[1], batch['max_length']))
        
        x_data = batch['x']

        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_reset = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']
        ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

        is_end_of_batch = False
        if numpy.sum(numpy.abs(x_reset)) < 1:
            # Print when we reach the end of an example (e.g. the end of a dialogue or a document)
            # Knowing when the training procedure reaches the end is useful for diagnosing training problems
            #print 'END-OF-BATCH EXAMPLE!'
            is_end_of_batch = True

        if commands:
            token_time = time.time()

        if state['use_nce']:
            y_neg = rng.choice(size=(10, max_length, x_data.shape[1]), a=model.idim, p=model.noise_probs).astype('int32')
            c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, y_neg, max_length, x_cost_mask, x_reset, ran_cost_utterance, ran_decoder_drop_mask)
        else:
            c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_cost_utterance, ran_decoder_drop_mask)

        total_token_time += token_time
        num_tokens_processed +=(batch['x'].shape[1] * batch['max_length'])

        print '%.3f words/s' % (num_tokens_processed/total_token_time)

        if commands:
            token_time = time.time()-token_time
            commands['timings'] = timings
            commands['token_time'] += token_time
            commands['num_tokens_processed'] += (batch['x'].shape[1] * batch['max_length'])

        # Print batch statistics
        print 'cost_sum', c
        print 'cost_mean', c / float(numpy.sum(x_cost_mask))
        print 'kl_divergence_cost_sum', kl_divergence_cost
        print 'kl_divergence_cost_mean', kl_divergence_cost / float(len(numpy.where(x_data == model.eos_sym)[0]))
        print 'posterior_mean_variance', posterior_mean_variance


        if numpy.isinf(c) or numpy.isnan(c):
            logger.warn("Got NaN cost .. skipping")
            gc.collect()
            continue

        train_cost += c
        train_kl_divergence_cost += kl_divergence_cost
        train_posterior_mean_variance += posterior_mean_variance

        train_done += batch['num_preds']
        train_dialogues_done += batch['num_dialogues']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            elapsed = this_time - start_time

            # Keep track of training cost for the last 'train_freq' batches.
            current_train_cost = train_cost/train_done

            if prev_train_done >= 1 and abs(train_done - prev_train_done) > 0:
                current_train_cost = float(train_cost - prev_train_cost)/float(train_done - prev_train_done)

            if numpy.isinf(c) or numpy.isnan(c):
                current_train_cost = 0

            prev_train_cost = train_cost
            prev_train_done = train_done

            h, m, s = ConvertTimedelta(this_time - start_time)

            # We need to catch exceptions due to high numbers in exp
            try:
                print ".. %.2d:%.2d:%.2d %4d mb # %d bs %d maxl %d acc_cost = %.4f acc_word_perplexity = %.4f cur_cost = %.4f cur_word_perplexity = %.4f acc_mean_word_error = %.4f acc_mean_kl_divergence_cost = %.8f acc_mean_posterior_variance = %.8f" % (h, m, s,\
                                 state['time_stop'] - (time.time() - start_time)/60.,\
                                 step, \
                                 batch['x'].shape[1], \
                                 batch['max_length'], \
                                 float(train_cost/train_done), \
                                 math.exp(float(train_cost/train_done)), \
                                 current_train_cost, \
                                 math.exp(current_train_cost), \
                                 float(train_misclass)/float(train_done), \
                                 float(train_kl_divergence_cost/train_done), \
                                 float(train_posterior_mean_variance/train_dialogues_done))
            except:
                pass

            #timings['train_progress'].append(math.exp(float(train_cost/train_done)))
            timings['train_progress'].append(math.exp(current_train_cost))

        ### Inspection phase
        # Evaluate gradient variance every 200 steps for GRU decoder
        if state['utterance_decoder_gating'].upper() == "GRU":
            if (step % 200 == 0) and (model.add_latent_gaussian_per_utterance):
                k_eval = 10

                softmax_costs = numpy.zeros((k_eval), dtype='float32')
                var_costs = numpy.zeros((k_eval), dtype='float32')
                gradients_wrt_softmax = numpy.zeros((k_eval, model.qdim_decoder, model.qdim_decoder), dtype='float32')
                for k in range(0, k_eval):
                    batch = add_random_variables_to_batch(model.state, model.rng, batch, None, False)
                    ran_cost_utterance = batch['ran_var_constutterance']
                    ran_decoder_drop_mask = batch['ran_decoder_drop_mask']
                    softmax_cost, var_cost, grads_wrt_softmax, grads_wrt_kl_divergence_cost = eval_grads(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_cost_utterance, ran_decoder_drop_mask)
                    softmax_costs[k] = softmax_cost
                    var_costs[k] = var_cost
                    gradients_wrt_softmax[k, :, :] = grads_wrt_softmax

                print 'mean softmax_costs', numpy.mean(softmax_costs)
                print 'std softmax_costs', numpy.std(softmax_costs)

                print 'mean var_costs', numpy.mean(var_costs)
                print 'std var_costs', numpy.std(var_costs)

                print 'mean gradients_wrt_softmax', numpy.mean(numpy.abs(numpy.mean(gradients_wrt_softmax, axis=0))), numpy.mean(gradients_wrt_softmax, axis=0)
                print 'std gradients_wrt_softmax', numpy.mean(numpy.std(gradients_wrt_softmax, axis=0)), numpy.std(gradients_wrt_softmax, axis=0)


                print 'std greater than mean', numpy.where(numpy.std(gradients_wrt_softmax, axis=0) > numpy.abs(numpy.mean(gradients_wrt_softmax, axis=0)))[0].shape[0]

                Wd_s_q = model.utterance_decoder.Wd_s_q.get_value()

                print 'Wd_s_q all', numpy.sum(numpy.abs(Wd_s_q)), numpy.mean(numpy.abs(Wd_s_q))
                print 'Wd_s_q latent', numpy.sum(numpy.abs(Wd_s_q[(Wd_s_q.shape[0]-state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :])), numpy.mean(numpy.abs(Wd_s_q[(Wd_s_q.shape[0]-state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :]))

                print 'Wd_s_q ratio', (numpy.sum(numpy.abs(Wd_s_q[(Wd_s_q.shape[0]-state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :])) / numpy.sum(numpy.abs(Wd_s_q)))

                if 'latent_gaussian_linear_dynamics' in state:
                    if state['latent_gaussian_linear_dynamics']:
                       prior_Wl_linear_dynamics = model.latent_utterance_variable_prior_encoder.Wl_linear_dynamics.get_value()
                       print 'prior_Wl_linear_dynamics', numpy.sum(numpy.abs(prior_Wl_linear_dynamics)), numpy.mean(numpy.abs(prior_Wl_linear_dynamics)), numpy.std(numpy.abs(prior_Wl_linear_dynamics))

                       approx_posterior_Wl_linear_dynamics = model.latent_utterance_variable_approx_posterior_encoder.Wl_linear_dynamics.get_value()
                       print 'approx_posterior_Wl_linear_dynamics', numpy.sum(numpy.abs(approx_posterior_Wl_linear_dynamics)), numpy.mean(numpy.abs(approx_posterior_Wl_linear_dynamics)), numpy.std(numpy.abs(approx_posterior_Wl_linear_dynamics))

                #print 'grads_wrt_softmax', grads_wrt_softmax.shape, numpy.sum(numpy.abs(grads_wrt_softmax)), numpy.abs(grads_wrt_softmax[0:5,0:5])
                #print 'grads_wrt_kl_divergence_cost', grads_wrt_kl_divergence_cost.shape, numpy.sum(numpy.abs(grads_wrt_kl_divergence_cost)), numpy.abs(grads_wrt_kl_divergence_cost[0:5,0:5])


        ### Evaluation phase
        if valid_data is not None and\
            step % state['valid_freq'] == 0 and step > 1:
                start_validation = True

        # Only start validation loop once it's time to validate and once all previous batches have been reset
        if start_validation and is_end_of_batch:
                start_validation = False
                valid_data.start()
                valid_cost = 0
                valid_kl_divergence_cost = 0
                valid_posterior_mean_variance = 0

                valid_wordpreds_done = 0
                valid_dialogues_done = 0


                logger.debug("[VALIDATION START]")

                while True:
                    batch = valid_data.next()

                    # Validation finished
                    if not batch:
                        break
                     
                    logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))
        
                    x_data = batch['x']
                    x_data_reversed = batch['x_reversed']
                    max_length = batch['max_length']
                    x_cost_mask = batch['x_mask']

                    x_reset = batch['x_reset']
                    ran_cost_utterance = batch['ran_var_constutterance']
                    ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

                    c, kl_term, c_list, kl_term_list, posterior_mean_variance = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask, x_reset, ran_cost_utterance, ran_decoder_drop_mask)

                    # Rehape into matrix, where rows are validation samples and columns are tokens
                    # Note that we use max_length-1 because we don't get a cost for the first token
                    # (the first token is always assumed to be eos)
                    c_list = c_list.reshape((batch['x'].shape[1],max_length-1), order=(1,0))
                    c_list = numpy.sum(c_list, axis=1)
                    
                    words_in_dialogues = numpy.sum(x_cost_mask, axis=0)
                    c_list = c_list / words_in_dialogues
                    

                    if numpy.isinf(c) or numpy.isnan(c):
                        continue
                    
                    valid_cost += c
                    valid_kl_divergence_cost += kl_divergence_cost
                    valid_posterior_mean_variance += posterior_mean_variance

                    # Print batch statistics
                    print 'valid_cost', valid_cost
                    print 'valid_kl_divergence_cost sample', kl_divergence_cost
                    print 'posterior_mean_variance', posterior_mean_variance


                    valid_wordpreds_done += batch['num_preds']
                    valid_dialogues_done += batch['num_dialogues']

                logger.debug("[VALIDATION END]") 
                 
                valid_cost /= valid_wordpreds_done
                valid_kl_divergence_cost /= valid_wordpreds_done
                valid_posterior_mean_variance /= valid_dialogues_done


                # We need to catch exceptions due to high numbers in exp
                try:
                    print "** valid cost (NLL) = %.4f, valid word-perplexity = %.4f, valid kldiv cost (per word) = %.8f, valid mean posterior variance (per word) = %.8f, patience = %d" % (float(valid_cost), float(math.exp(valid_cost)), float(valid_kl_divergence_cost), float(valid_posterior_mean_variance), patience)
                except:
                    try:
                        print "** valid cost (NLL) = %.4f, patience = %d" % (float(valid_cost), patience)
                    except:
                        pass


                timings["train_cost"].append(train_cost/train_done)
                timings["train_kl_divergence_cost"].append(train_kl_divergence_cost/train_done)
                timings["train_posterior_mean_variance"].append(train_posterior_mean_variance/train_dialogues_done)
                timings["valid_cost"].append(valid_cost)
                timings["valid_perplexity"].append(float(math.exp(valid_cost)))
                timings["valid_kl_divergence_cost"].append(valid_kl_divergence_cost)
                timings["valid_posterior_mean_variance"].append(valid_posterior_mean_variance)


                if (len(timings["valid_cost"]) == 0) \
                    or (valid_cost < numpy.min(timings["valid_cost"])) \
                    or (save_model_on_first_valid and valid_rounds == 0):
                    patience = state['patience']

                    # Save model if there is  decrease in validation cost
                    if commands:
                        save2(model, timings, commands)
                    else:
                        save(model, timings)
                    print 'best valid_cost', valid_cost
                elif valid_cost >= timings["valid_cost"][-1] * state['cost_threshold']:
                    patience -= 1

                if args.save_every_valid_iteration:
                    if commands:
                        save2(model, timings, commands)
                    else:
                        save(model, timings, '_' + str(step) + '_')
                if args.auto_restart:
                    if commands:
                        save2(model, timings, commands)
                    else:
                        save(model, timings, '_auto_')


                # Reset train cost, train misclass and train done metrics
                train_cost = 0
                train_done = 0
                prev_train_cost = 0
                prev_train_done = 0

                # Count number of validation rounds done so far
                valid_rounds += 1

        step += 1

    logger.debug("All done, exiting...")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")

    parser.add_argument("--force_train_all_wordemb", action='store_true', help="If true, will force the model to train all word embeddings in the encoder. This switch can be used to fine-tune a model which was trained with fixed (pretrained)  encoder word embeddings.")

    parser.add_argument("--save_every_valid_iteration", action='store_true', help="If true, will save a unique copy of the model at every validation round.")

    parser.add_argument("--auto_restart", action='store_true', help="If true, will maintain a copy of the current model parameters updated at every validation round. Upon initialization, the script will automatically scan the output directory and and resume training of a previous model (if such exists). This option is meant to be used for training models on clusters with hard wall-times. This option is incompatible with the \"resume\" and \"save_every_valid_iteration\" options.")

    parser.add_argument("--prototype", type=str, help="Prototype to use (must be specified)", default='prototype_state')

    parser.add_argument("--reinitialize-latent-variable-parameters", action='store_true', help="Can be used when resuming a model. If true, will initialize all latent variable parameters randomly instead of loading them from previous model.")

    parser.add_argument("--reinitialize-decoder-parameters", action='store_true', help="Can be used when resuming a model. If true, will initialize all parameters of the utterance decoder randomly instead of loading them from previous model.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Models only run with float32
    assert(theano.config.floatX == 'float32')

    args = parse_args()
    train(args)
