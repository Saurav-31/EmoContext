#Please use python 3.5 or above
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from lstm_model import buildModel
from utils import *
import json, argparse, os
import re
import io
import sys

# Path to training and testing data file. This data can be downloaded from a link, details of which will be provided.
trainDataPath = ""
testDataPath = ""
# Output file that will be generated. This file can be directly submitted.
solutionPath = ""
# Path to directory where GloVe file is saved.
gloveDir = ""

NUM_FOLDS = None                   # Value of K in K-fold Cross Validation
NUM_CLASSES = None                 # Number of classes - Happy, Sad, Angry, Others
MAX_NB_WORDS = None                # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer 
MAX_SEQUENCE_LENGTH = None         # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = None               # The dimension of the word embeddings
BATCH_SIZE = None                  # The batch size to be chosen for training the model.
LSTM_DIM = None                    # The dimension of the representations learnt by the LSTM model
DROPOUT = None                     # Fraction of the units to drop for the linear transformation of the inputs. Ref - https://keras.io/layers/recurrent/
NUM_EPOCHS = None                  # Number of epochs to train a model for


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}


def main():
    parser = argparse.ArgumentParser(description="Baseline Script for SemEval")
    parser.add_argument('-config', help='Config to read details', required=True)
    args = parser.parse_args()

    with open(args.config) as configfile:
        config = json.load(configfile)
        
    global trainDataPath, testDataPath, solutionPath, gloveDir
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE    
    
    trainDataPath = config["train_data_path"]
    testDataPath = config["test_data_path"]
    solutionPath = config["solution_path"]
    gloveDir = config["glove_dir"]
    
    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]
    threshold = 5

    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(trainDataPath, mode="train")
    trainTexts, word_counts = preprocess_fn(trainTexts)
    # Write normalised text to file to check if normalisation works. Disabled now. Uncomment following line to enable   
    # writeNormalisedData(trainDataPath, trainTexts)
    print("Processing test data...")
    testIndices, testTexts = preprocessData(testDataPath, mode="test")
    testTexts, word_counts_test = preprocess_fn(testTexts)

    # writeNormalisedData(testDataPath, testTexts)

    print("Extracting tokens...")
    #tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    #tokenizer.fit_on_texts(trainTexts)
    #trainSequences = tokenizer.texts_to_sequences(trainTexts)
    #testSequences = tokenizer.texts_to_sequences(testTexts)

    #wordIndex = tokenizer.word_index
    #print("Found %s unique tokens." % len(wordIndex))

    #print("Populating embedding matrix...")
    #embeddingMatrix = getEmbeddingMatrix(wordIndex)
    embeddingMatrix, vocab_to_int = getEmbeddingMatrix(word_counts, threshold, gloveDir, EMBEDDING_DIM)

    trainSequences = final_fn(vocab_to_int, word_counts, trainTexts)
    testSequences = final_fn(vocab_to_int, word_counts, testTexts)

    data = pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)
        
    # Randomize data
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]
      
    # Perform k-fold cross validation
    metrics = {"accuracy" : [],
               "microPrecision" : [],
               "microRecall" : [],
               "microF1" : []}
    
    
    print("Starting k-fold cross validation...")
    for k in range(NUM_FOLDS):
        print('-'*40)
        print("Fold %d/%d" % (k+1, NUM_FOLDS))
        validationSize = int(len(data)/NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k+1)
            
        xTrain = np.vstack((data[:index1],data[index2:]))
        yTrain = np.vstack((labels[:index1],labels[index2:]))
        xVal = data[index1:index2]
        yVal = labels[index1:index2]
        print("Building model...")
        model = buildModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE)
        model.fit(xTrain, yTrain, 
                  validation_data=(xVal, yVal),
                  epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

        predictions = model.predict(xVal, batch_size=BATCH_SIZE)
        accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
        metrics["accuracy"].append(accuracy)
        metrics["microPrecision"].append(microPrecision)
        metrics["microRecall"].append(microRecall)
        metrics["microF1"].append(microF1)
        
    print("\n============= Metrics =================")
    print("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"])/len(metrics["accuracy"])))
    print("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"])/len(metrics["microPrecision"])))
    print("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"])/len(metrics["microRecall"])))
    print("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"])/len(metrics["microF1"])))
    
    print("\n======================================")
    
    print("Retraining model on entire data to create solution file")
    model = buildModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE)
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))
    # model = load_model('EP%d_LR%de-5_LDim%d_BS%d.h5'%(NUM_EPOCHS, int(LEARNING_RATE*(10**5)), LSTM_DIM, BATCH_SIZE))

    print("Creating solution file...")
    testData = pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(testData, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(solutionPath, "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')        
        with io.open(testDataPath, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')
    print("Completed. Model parameters: ")
    print("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" 
          % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE))
    
               
if __name__ == '__main__':
    main()