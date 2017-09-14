from spell_correction import get_spell_corrector
from nltk import word_tokenize
from data.encoding_tools import encode
import scipy
import string
class TextPipeline():

    def __init__(self, model_manager, language = 'dutch'):
        self.model_manager = model_manager
        self.language = language
        self.components = []

    def add_tokenizer_sentence(self, priority=1):

        def tokenize(input):
            return word_tokenize(input, language=self.language)

        self.components.append((priority, tokenize))


    def add_sentence_splitter(self, priority=1):

        def split(input):
            return input.split(' ')

        self.components.append((priority, split))


    def add_spell_correction(self, priority = 2):
        corrector = get_spell_corrector(self.model_manager)

        def check_tokenized(input):
            output = []
            for token in input:
                output.append(corrector(token))

            return output

        self.components.append((priority, check_tokenized))

    def add_customer_token(self, priority = 4):

        def adder(input):
            assert type(input) == list

            input.insert(0, '<customer>')
            return input

        self.components.append((priority, adder))

    def add_pretty_output_merger(self):

        def merger(input):
            if type(input) is str or type(input) is basestring:
                return input

            return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in input]).strip()

        self.components.append((100, merger))


    def add_encoder_component(self, priority = 3):

        if not hasattr(self, 'encoder'):
            self.encoder = self.model_manager.load_currently_selected_model()

        encoder = self.encoder

        def enc(input):

            indices = []
            for token in input:
                if token in encoder.str_to_idx:
                    indices.append(encoder.str_to_idx[token])
                else:
                    indices.append(0)

            eou = encoder.str_to_idx['</u>']
            if indices[-1] is not eou:
                indices.append(eou)

            feature_vec = encode(indices, encoder, as_text=False)

            return input, feature_vec[0][0][0], feature_vec[1][0][0]

        self.components.append((priority, enc))


    def process(self, raw_chat_input):

        input = raw_chat_input
        for priority, component in sorted(self.components, key= lambda pair:pair[0]):

            output = component(input)
            input = output

        return output


def intent_debug(model_manager, language = 'dutch'):

    intents = {}
    desc = []
    intent = None
    for line in open('./data/intents.txt', 'rb'):
        line = line.strip()
        print line
        if line.startswith('@'):
            if intent:
                intents[intent] = desc
                desc = []

            intent = line[1:]
        elif len(line) > 0:
            desc.append(line)
        else:
            continue

    if desc:
        intents[intent] = desc

    preprocessor = TextPipeline(model_manager, language)
    preprocessor.add_tokenizer_sentence()
    preprocessor.add_spell_correction()
    preprocessor.add_customer_token(5)
    preprocessor.add_encoder_component(6)


    encodings = {}

    for intent, desc in intents.iteritems():

        meanings = []
        for user_says in desc:
            output, dia_emb, utt_emb = preprocessor.process(user_says)

            meanings.append((user_says, utt_emb))

        encodings[intent] = meanings

    metric = scipy.spatial.distance.cosine
    while 1:

        input = raw_input('Find similar -')
        output, dia_emb, utt_emb = preprocessor.process(input)


        intent_scores = []
        for intent, embeddings in encodings.iteritems():

            for user_says, embedding in embeddings:

                score = metric(utt_emb, embedding)
                intent_scores.append((score, intent, user_says))


        for scored in sorted(intent_scores, key=lambda obj: obj[0], reverse=True):
            print scored






if __name__ == '__main__':
    from model_manager import ModelManager

    m = ModelManager('vodafone_hred_v3')
    intent_debug(m)
    exit()



    chat = TextPipeline(m)
    chat.add_sentence_splitter()
    #chat.add_tokenizer_sentence()
    #chat.add_spell_correction()
    chat.add_encoder_component()

    output = chat.process('<assistant> hoi r de bel , je chat met <first_name> . waarmee kan ik je helpen ?')

    test_str = "<assistant> hoi r de bel , je chat met <first_name> . waarmee kan ik je helpen ? </u> </s> <customer> hallo <first_name> ik snap er niks van ik ben <number> dag in usa en heb mijn mobiele netwerk uit staan en ineens een bericht over <number> % ober mijn data bundel </u> <number> sec later <number> % </u> ik zit nu met een rekening terwijl ik naar mijn weten niks heb gedaan </u> </s> <assistant> om je verder te helpen heb ik de volgende gegevens nodig : - je voorletters , achternaam + geboortedatum - je huisnummer + postcode - de laatste <number> cijfers van je bankrekeningnummer . </u> </s> <customer> r de bel <date> <number> <post_code> <number> </u> of <unk> dat is de postbus </u> </s> <assistant> wat ik kan zien is dat jou data roaming aan staat </u> en je hebt geen bundel aan staan voor america dit betekend dat je <decimal> ex btw per mb betaald </u> en daarom heb jij nu je data limiet van <number> euro verbruikt </u> </s> <customer> maar als ik mijn mobiele data uitzet dan staat data roaming toch ook uit want hij verdwijnt uit mijn beeld kan ik een foto sturen ? </u> om te laten zien wat ik bedoel </u> hij staat uit naar mijn weten </u> </s> <assistant> je hebt een internet <unk> gemaakt </u> anders haden wij niet kunnen zien dat je in america was </u> </s> <customer> pfffffff wat een ellende </u> </s> <assistant> ik zou de volgende keer voor dat je op vakantie gaat even na vraag doen bij ons dat had je in dit geval <decimal> kunnen schelen : ( </u> </s> <customer> ik heb <number> min aangezet omdat ik de weg kwijt was kan dat echt zo snel gaan ? </u> </s> <assistant> als je <decimal> per mb betaald inc btw dan gaat dat heel snel </u> zijn maar <number> mb die je gebruikt hebt </u> </s> <customer> kan wel janken nu kan er niks meer gebeuren toch ? </u> tm wanneer niet ? zit hier nog een paar dagen </u> </s> <assistant> nee klopt nu hebben wij er een blokade op gezet </u> we kunnen hem deblokeren als je wil </u> maar dan kunnen er wel hogere kosten komen </u> </s> <customer> nee niet deblokkeren maar tot wanneer zit de blokkade erop ? </u> ik ben hier nog effe </u> </s> <assistant> tm <date> </u> </s> <customer> dan ben ik terug </u> </s> <assistant> oke </u> </s> <customer> wat nou als ik in geval van nood wel internet nodig heb ? kan ik nu mijn blox wereld aanzetten en daar begrijp van maken of ziet hij dat ook als extra buiten je bundel ? </u> </s> <assistant> dan moeten we de blokade ook verwijderen </u> anders kom je niet op internet </u> </s> <customer> oke bedankt voor de informatie </u> </s> <assistant> graag gedaan </u> geniet er van in america ! ! </u> ; ) </u> </s> <customer> thanks </u>"


    encoder = m.load_currently_selected_model()

    true_result = encode(test_str, encoder)




    print





