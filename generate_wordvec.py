import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from nltk.tokenize.casual import TweetTokenizer
from utils import *

#book_filenames = sorted(glob.glob("train.txt"))
trainIndices, trainTexts, labels = preprocessData("train.txt", mode="train")
raw_sentences, word_counts = preprocess_fn(trainTexts)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(raw_sentences[:10])
print(sentence_to_wordlist(raw_sentences[2]))


token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))


num_features = 100
min_word_count = 2
context_size = 5
downsampling = 1e-3
seed = 1

model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

model.build_vocab(sentences)

print("Word2Vec vocabulary length:", len(model.wv.vocab))

model.train(sentences, epochs=model.iter, total_examples=model.corpus_count)

if not os.path.exists("trained"):
    os.makedirs("trained")

model.save(os.path.join("trained", "model.w2v"))

model = w2v.Word2Vec.load(os.path.join("trained", "model.w2v"))

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = model.wv.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model.wv.vocab[word].index])
            for word in model.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

points.head(10)
sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))

model.most_similar("ridiculous")

'''
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = model.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2
'''


