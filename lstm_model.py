from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers

def buildModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE):
    """Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = Embedding(embeddingMatrix.shape[0],
                                EMBEDDING_DIM,
                                weights=[embeddingMatrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    model = Sequential()
    model.add(embeddingLayer)
    model.add(LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    rmsprop = optimizers.rmsprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model