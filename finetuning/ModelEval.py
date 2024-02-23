from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

from nltk.translate.meteor_score import single_meteor_score

from rouge_score import rouge_scorer

import spacy

import numpy as np

 

def calculate_bleu(reference, candidate):

    return sentence_bleu([reference.split()], candidate.split())

 

def calculate_corpus_bleu(references, candidates):

    return corpus_bleu([[ref.split()] for ref in references], [candidate.split() for candidate in candidates])

 

def calculate_meteor(reference, candidate):

    return single_meteor_score(reference, candidate)

 

def calculate_rouge(reference, candidate):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = scorer.score(reference, candidate)

    return scores['rougeL'].fmeasure

 

def calculate_spacy_similarity(reference, candidate):

    nlp = spacy.load('en_core_web_sm')

    reference_doc = nlp(reference)

    candidate_doc = nlp(candidate)

    return reference_doc.similarity(candidate_doc)

 

def evaluate_model(references, candidates):

    bleu_scores = [calculate_bleu(ref, cand) for ref, cand in zip(references, candidates)]

    corpus_bleu_score = calculate_corpus_bleu(references, candidates)

    meteor_scores = [calculate_meteor(ref, cand) for ref, cand in zip(references, candidates)]

    rouge_scores = [calculate_rouge(ref, cand) for ref, cand in zip(references, candidates)]

    spacy_similarities = [calculate_spacy_similarity(ref, cand) for ref, cand in zip(references, candidates)]

 

    average_bleu = np.mean(bleu_scores)

    average_meteor = np.mean(meteor_scores)

    average_rouge = np.mean(rouge_scores)

    average_spacy_similarity = np.mean(spacy_similarities)

 

    print(f'Average BLEU Score: {average_bleu}')

    print(f'Corpus BLEU Score: {corpus_bleu_score}')

    print(f'Average METEOR Score: {average_meteor}')

    print(f'Average ROUGE Score: {average_rouge}')

    print(f'Average SpaCy Similarity: {average_spacy_similarity}')

 

# Example usage

references = ["This is a reference sentence."]

candidates = ["This is a generated sentence."]

 

evaluate_model(references, candidates)


 

pip install nltk rouge-score spacy

python -m spacy download en_core_web_sm