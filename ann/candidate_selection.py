
from scipy.spatial.distance import cosine
def cosine_similarity(vec1, vec2):
    assert len(vec1) == len(vec2)
    return cosine(vec1, vec2)

def context_relevance(candidates):
    pass

def answer_relevance(search_context):

    instances = []
    for label in search_context['labels']:
        utt_emb = search_context['utterance_embeddings']
        instances.append((label, utt_emb[label[0]][label[1]]))

    scored = []
    for label, utt_emb in instances:

        score = 0

        for label2, utt_emb2 in instances:
            score += cosine(utt_emb, utt_emb2)

        scored.append((score, label))

    return scored