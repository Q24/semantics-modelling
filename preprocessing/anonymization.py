from dateutil.parser import parse
import datetime
import os
import time
import pickle
import utils
# preprocess anonymization.anonymization anonymize

valid_name_tags = ['nounsg']

invalid_first_names = ['just', 'line', 'take', 'koop', 'hope', 'long', 'door', 'alle']

valid_street_name_endings = ['straat', 'plein', 'gracht', 'poort', 'laan', 'singel', 'grachtje', 'st.']

month_names = ['<month>', 'januari', 'january', 'februari', 'february', 'maart', 'march',
               'april', 'mei', 'juni', 'june', 'juli', 'augustus', 'august', 'september', 'oktober',
               'october', 'november', 'december']



valid_date_formats = ['%Y-%m-%d', '%Y/%m/%d',
                      '%d-%m-%Y', '%d/%m/%Y',
                      '%m-%d-%Y', '%m/%d/%Y']

class FilterRule():

    def __init__(self, num_input_tokens):
        self.N = num_input_tokens
        self.count = 0




    def filter_city_names(self, input, tag):
        if input in self.data['cities'] and len(input) > 4 and tag in valid_name_tags:
            input = '<city>'
            self.count += 1
        return input

    def filter_first_names(self, input, tag):
        if input in self.data['first_names'] and len(input) > 3 and 'noun' in tag and input not in invalid_first_names:
            #if input in self.data['dutch_words']:
            #    print 'common word ' + self.data['dutch_words'][input]

            input = '<first_name>'
            self.count += 1
        return input

    def filter_dates(self, input, tag):
        if input in month_names and tag == 'nounprop':
            input = '<month>'
            self.count += 1
        return input

    #3
    def filter_dates_times_numbers(self, inputs, tags):

        properties = {0:[], 1:[], 2:[]}

        for idx, token in enumerate(inputs):
            if is_decimal(token):
                properties[idx].append('is_decimal')

            if is_digit(token):
                properties[idx].append('is_digit')

                token_int = int(token)

                if token_int > 1800 and token_int < 2100:
                    properties[idx].append('is_valid_year')

                if 1 <= token_int and token_int <= 12:
                    properties[idx].append('is_valid_month')

                if 1 <= token_int and token_int <= 31:
                    properties[idx].append('is_valid_day')


            if token.isalpha():
                properties[idx].append('is_only_text')

                if token in month_names:
                    properties[idx].append('is_month_str')



            if ':' in token:
                try:
                    if time.strptime(token, '%H:%M'):
                        properties[idx].append('is_time')
                except:
                    pass

            if is_date(token):
                properties[idx].append('is_date')

        values = properties.values()

        flattened_properties = [item for sublist in values for item in sublist]

        if 'is_valid_year' in flattened_properties and all(len(sublist) > 0 for sublist in values):
            try:
                dateobj = parse('-'.join(inputs))
            except:
                dateobj = None

            if dateobj:
                if 'is_valid_year' in flattened_properties:
                    for idx, props in enumerate(values):
                        if 'is_valid_year' in props:
                            year_loc = idx
                            break

                    if year_loc == 0:
                        output = ['<year>', '<month>', '<day>']
                    if year_loc == 1:
                        output = ['<month>', '<year>', '<day>']
                    if year_loc == 2:
                        output = ['<day>', '<month>', '<year>']

                    return output

        if all(len(sublist) > 0 for sublist in values):
            prop1 = 'is_digit' in properties[0]
            prop2 = 'is_month_str' in properties[1]
            prop3 = 'is_digit' in properties[2]

            if prop1 and prop2 and prop3:
                return ['<day>', '<month>', '<year>']

        for idx, (token, props) in enumerate(zip(inputs, values)):
            if 'is_time' in props:
                inputs[idx] = '<time>'

            if 'is_date' in props:
                inputs[idx] = '<date>'

            if 'is_decimal' in props:
                inputs[idx] = '<decimal>'

        return inputs

    # 2 tokens
    def filter_last_names(self, input_tokens, tags):
        if '<first_name>' in input_tokens:
            if input_tokens[1] in self.data['last_names'] or input_tokens[0] in self.data['last_names']:
                print input_tokens
                print

        return input_tokens

    # 2 tokens
    def filter_text_combined_with_numbers(self, input, tags):
        digits = 0
        characters = 0


        for char in input:
            if char in "0123456789":
                digits += 1
            else:
                characters += 1

        if digits == 0:
            return input

        self.count += 1

        if characters == 0:
            token_int = int(input)
            if token_int > 1800 and token_int < 2100:
                return '<year>'

            return '<number>'


        if input in self.data['post_codes']:
            return '<post_code>'

        return '<digits+text>'

    # 3
    def filter_street_names(self, input_tokens, tags):


        indices = [[0,1,2], [0,1], [1,2], [1], [2], [0]]
        combinations = [''.join([input_tokens[idx] for idx in indic]) for indic in indices]

        for combination, indic in zip(combinations, indices):

            if not any(combination.endswith(ending) for ending in valid_street_name_endings):
                continue

            if not combination in self.data['street_names']:
                continue

            if len(combination) == 1 and combination[0] in valid_street_name_endings:
                continue
            for idx in indic:
                input_tokens[idx] = '<street_name>'

            break


        return input_tokens



def init_static_filtering():
    cities, post_codes, street_names, first_names, prefixes, last_names, dutch_words = load_dictionaries()

    data = {'cities': cities, 'post_codes': post_codes, 'street_names': street_names, 'first_names': first_names,
            'prefixes': prefixes, 'last_names': last_names, 'dutch_words': dutch_words}

    for filter_rule in filter_rules:
        filter_rule.data = data

class Anonymizer:

    def __init__(self):
        self.cities, self.post_codes, self.street_names, self.first_names, self.prefixes, self.last_names, self.dutch_words = load_dictionaries()

        self.data = {'cities': self.cities, 'post_codes': self.post_codes, 'street_names': self.street_names, 'first_names': self.first_names,
            'prefixes': self.prefixes, 'last_names': self.last_names, 'dutch_words': self.dutch_words}

        with open('./preprocessing/dutch.pickle', 'rb') as f:
            self.tagger = pickle.load(f)

        for filter_rule in filter_rules:
            filter_rule.data = self.data

    def anonymize(self,iterable_dialogues):

        anonymized = []

        dialogues = iterable_dialogues[:]


        for prog, dialogue in enumerate(dialogues):

            str = dialogue.replace('</s>', '.').replace('</u>', '.')

            dialogue_tagged = self.tagger.tag(str.split())


            #anonymized.append(dialogue)

            anonymized_tokens = self.anonymize_dialouge(dialogue_tagged)

            original_tokens = dialogue.split()

            assert len(anonymized_tokens) == len(original_tokens)

            for idx in xrange(len(anonymized_tokens)):
                an_token = anonymized_tokens[idx][0]

                if an_token != '.':
                    original_tokens[idx] = an_token

            anonymized.append(original_tokens)

            #filter_progress = '%d dialogues, %d cities, %d street_names, %d first_names, %d last_names'%(prog+1, filter_cn.count, filter_sn.count, filter_fn.count, filter_ln.count)

            #utils.print_progress_bar(prog+1, len(dialogues), tokens=50, additional_text=filter_progress)

        return anonymized






    def anonymize_dialouge(self, dialogue_tokens):

        meta_tokens = []

        # store meta tokens to ensure that they won't be overwritten
        for idx, (token, tag) in enumerate(dialogue_tokens):
            if tag == 'meta':
                meta_tokens.append((idx, token))

        for filter_rule in filter_rules:

            # how many tokens the rule processes at once
            token_span = filter_rule.N

            for idx in xrange(len(dialogue_tokens)-token_span+1):

                indices = range(idx, idx+token_span)

                tokens = [dialogue_tokens[idx][0].lower() for idx in indices]
                tags = [dialogue_tokens[idx][1] for idx in indices]

                # ignore already replaced tokens and meta-information tokens such as </s>
                # tokens_with_brackets = [token for token in tokens if token.endswith('>') and token.startswith('<')]

                if len(tokens) == 1:
                    filtered_tokens = filter_rule.filter(filter_rule, tokens[0], tags[0])
                    filtered_tokens = [filtered_tokens]
                else:
                    filtered_tokens = filter_rule.filter(filter_rule, tokens, tags)

                if filtered_tokens == None:
                    continue

                for small_idx, token_idx in enumerate(indices):
                    #if dialogue_tokens[token_idx] != filtered_tokens[small_idx]:
                    #if '<street_name>' == filtered_tokens[small_idx]:
                        #print dialogue_tokens[token_idx-1], dialogue_tokens[token_idx]
                        #print '%s replaced by %s at %d'%(dialogue_tokens[token_idx], filtered_tokens[small_idx], token_idx)

                    dialogue_tokens[token_idx] = (filtered_tokens[small_idx], tags[small_idx])

        for idx, token in meta_tokens:
            dialogue_tokens[idx] = (token, 'meta')

        return dialogue_tokens



def has_any_digit(str):
    for char in str:
        if char.isdigit():
            return True

    return False


def is_date(str):
    if not ('-' in str or '/' in str):
        return False

    for date_format in valid_date_formats:
        try:
            datetime.datetime.strptime(str, date_format)
            return True
        except:
            continue

    return False

def is_decimal(str):
    if ',' in str:
        split = str.split(',')
    elif '.' in str:
        split = str.split('.')
    else:
        return False

    if len(split) != 2:
        return False

    if split[0].isdigit() and split[1].isdigit():
        return True

    return False


def is_digit(str):
    for char in str:
        if char not in "0123456789":
            return False

    return True

def load_dictionaries(**kwargs):

    address_loc = kwargs.get('address_file', './preprocessing/ref_postcode_export_gemini.20161012_220243')
    first_names_loc = kwargs.get('first_names_file', './preprocessing/voornamentop10000.xml')
    last_names_loc = kwargs.get('last_names_file', './preprocessing/fn_10kw.xml')
    dutch_words = kwargs.get('dictionary_file', './preprocessing/dutch.dict.txt')

    cities = set()
    post_codes = set()
    street_names = set()

    with open(address_loc, 'rb') as file:
        for line in file.readlines():

            instance = line.replace(os.linesep, '').split(';')

            # add city
            cities.add(instance[5].lower())

            # add post code
            post_codes.add(instance[0].lower())

            # add street name
            street_names.add(instance[4].lower())

    #print 'maastricht' in cities
    #print 'cities: ', len(cities)
    #print 'post_codes: ', len(post_codes)
    #print 'street_names: ', len(street_names)

    from xml.etree import ElementTree
    from lxml import etree

    with open(first_names_loc, 'rb') as file:
        schema_root = etree.XML(file.read())

    first_names = set()

    lengths = []

    for first_name in schema_root.findall('.//voornaam'):
        first_names.add(first_name.text.lower())
        lengths.append(len(first_name.text))


    #print 'first_names: ', len(first_names)
    #print 'min: ', min(lengths)
    #print 'max: ', max(lengths)
    #print 'avg:', (sum(lengths)/float(len(lengths)))




    with open(last_names_loc, 'rb') as file:
        schema_root = etree.XML(file.read())

    prefixes = set([prefix.text.lower() for prefix in schema_root.findall('.//prefix') if type(prefix.text) is str])

    last_names = set([prefix.text.lower() for prefix in schema_root.findall('.//naam')])

    #print 'prefixes: ', len(prefixes)
    #print 'last_names: ', len(last_names)


    if os.path.exists(dutch_words):

        with open(dutch_words, 'rb') as file:
            lines = file.readlines()

        dutch_words = {}

        for line in lines:
            split = line.split('/')
            if len(split) != 2:
                continue

            word = split[0].lower()

            if any(i.isdigit() for i in word) or len(word) <= 1:
                continue

            dutch_words[word] = split[1]

        #print 'dutch_words: ', len(dutch_words)
    else:
        dutch_words = []

    return cities, post_codes, street_names, first_names, prefixes, last_names, dutch_words


filter_cn = FilterRule(1)
filter_cn.filter = FilterRule.filter_city_names

filter_da = FilterRule(1)
filter_da.filter = FilterRule.filter_dates

filter_nm = FilterRule(3)
filter_nm.filter = FilterRule.filter_dates_times_numbers

filter_fn = FilterRule(1)
filter_fn.filter = FilterRule.filter_first_names

filter_ln = FilterRule(2)
filter_ln.filter = FilterRule.filter_last_names

filter_tn = FilterRule(1)
filter_tn.filter = FilterRule.filter_text_combined_with_numbers

filter_sn = FilterRule(3)
filter_sn.filter = FilterRule.filter_street_names

filter_rules = [#filter_cn,
                filter_da,
                filter_nm,
                filter_fn,
                #filter_ln,
                filter_tn,
                filter_sn
                ]


if __name__ == '__main__':
    import os
    file_loc = '../models/vodafone/processed_data/lines.tokenized.txt'
    with open(file_loc, 'rb') as file:
        dialogues = file.readlines()

    dialogues = anonymize(dialogues)

    save_to = './data/anonymized.txt'
    with open(save_to, 'wb') as save:

        for line in dialogues:
            save.write(line)
            save.write(os.linesep)



