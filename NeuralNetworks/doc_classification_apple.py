from commons.dataload_utils.load_apple import load_apple
from commons.text_utils.create_dictionary import create_dictionary
from commons.datatransform_utils.one_hot_str import one_hot_str
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np


if __name__ == '__main__':

    # define documents
    company_docs, fruit_docs, company_tokens, fruit_tokens, max_doc_len = load_apple()

    print(company_docs[:5])
    print(fruit_docs[:5])

    # define class labels
    # 0              # 1
    labels_names = ['computer-company', 'fruit']
    labels = np.array([0] * len(company_docs) + [1] * len(fruit_docs))
    labels = np.reshape(labels, (labels.shape[0], 1))

    docs = company_docs + fruit_docs

    print('Document number: {}\tCompany: {} - Fruit: {}'.format(len(labels), len(company_docs), len(fruit_docs)))

    my_dictionary, popular_words, unpopular_words = create_dictionary(company_tokens + fruit_tokens, 4, 400, 30, 30)
    print("Dictionary size: {}".format(len(my_dictionary)))
    print("Most popular words:\n{}".format(popular_words))
    print("Least popular words:\n{}\n".format(unpopular_words))

    # One-hot encoding and padding
    vocab_size = len(my_dictionary)
    encoded_docs = [one_hot_str(my_dictionary, d, False) for d in docs]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_doc_len, padding='post')

    X = padded_docs
    Y = labels
    print("Shape of X: {}".format(X.shape))
    print("Shape of Y: {}".format(Y.shape))

    ## Train model
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_doc_len))  # 8-embedding
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # Fit the model
    model.fit(X, Y, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(X, Y, verbose=0)
    print('\nTrain accuracy: %f' % (accuracy * 100))

    # Predict
    sample_sentence = 'The world crop of apples averages more than 60 million metric tons a year. Of the American crop, more than half is normally used as fresh fruit.'
    print('Predicting for sample sentence about the apple fruit: "' + sample_sentence + '"')
    X_test = [one_hot_str(my_dictionary, sample_sentence, False)]
    X_test = pad_sequences(X_test,  maxlen=max_doc_len, padding='post')
    pred = model.predict(X_test)
    print("Prediction: {}, actual value: {}".format(np.squeeze(pred), 1))
