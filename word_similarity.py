import io, os
import numpy as np

'''
gloveDir = "gloveVec/"
embeddingsIndex = {}

with io.open(os.path.join(gloveDir, 'glove.6B.100d.txt'), encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddingVector = np.asarray(values[1:], dtype='float32')
        embeddingsIndex[word] = embeddingVector
'''

from gensim.models import KeyedVectors
path = "/mnt/sqnap1/devraj/word2vec/GoogleNews-vectors-negative300.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

model.similarity('france', 'spain')
model.most_similar("hate")