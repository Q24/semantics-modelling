import os
import sys
import datetime
from time import time

def create_folder_if_not_existing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)




def print_progress_bar(current_progress, total_progress, done = False, additional_text = '', tokens=25, start_time = None):
    sys.stdout.write('\r')
    if done:
        sys.stdout.write('[' + str('#' * tokens) + ('_' * 0) + ']' + '100.00% ' + additional_text + os.linesep)
        return

    if total_progress <= 0 or current_progress < 0:
        return
    percentage = float(current_progress)/float(total_progress)
    num_progress_tokens = int(tokens* percentage)
    empty_tokens = tokens-num_progress_tokens
    percentage_str = ' %.2f%%'%(percentage*100)
    sys.stdout.write('['+ str('#'*num_progress_tokens) + ('_'*empty_tokens) + ']' + percentage_str)

    if additional_text:
        sys.stdout.write(' ' +additional_text)

    if start_time is not None:
        elapsed_time = time()-start_time
        remaining_time = elapsed_time*((1.-percentage)/percentage)

        remaining_time = str(datetime.timedelta(seconds=remaining_time)).split('.')[0]

        sys.stdout.write(' remaining time: %s' %remaining_time)

    if current_progress == total_progress:
        sys.stdout.flush()



def batch_iterator(single_yield_iterator, batch_size, apply_on_element = None):

    batch = []

    for element in single_yield_iterator:

        if apply_on_element:
            element = apply_on_element(element)

        batch.append(element)

        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch