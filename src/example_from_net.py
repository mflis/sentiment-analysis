from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential

from src.helpers import *


# source
#  https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(VOCABULARY_LIMIT, input_dim=VOCABULARY_LIMIT,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


tokenizer = get_tokenizer()
columns = getColumns(dataPath(), ROW_LIMIT)
texts, scores = columns
scores_array = np.reshape(np.asarray(scores), (-1, 1))
tokenizer.fit_on_texts(texts)
x_train = tokenizer.texts_to_matrix(texts, mode='tfidf')

# evaluate model with standardized dataset
estimator = create_baseline()
estimator.fit(x_train, scores_array, batch_size=4096, epochs=2)
# 1+1
# roc_auc_score(scores_array, x_train)
