'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from src import ROOT_DIR
from src.custom_metrics import *
from src.helpers import *

BASE_DIR = os.path.join(ROOT_DIR, 'data')
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
TEXT_DATA_FILE = os.path.join(BASE_DIR, 'AmazonReviews-raw.csv')
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

# second, prepare text samples and their labels
print('Processing text dataset')

labels_index = {}  # dictionary mapping label name to numeric id
texts, labels_list = getColumns(TEXT_DATA_FILE, rows_cut=1000)
labels = labels_list
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED)
print('Preparing embedding matrix.')
# todo rewrite more efiicient
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
# x = Conv1D(128, 5, activation='relu')(embedded_sequences)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = MaxPooling1D(5)(x)
# x = Conv1D(128, 5, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
# x = Dense(128, activation='relu')(x)
# preds = Dense(1, activation='sigmoid')(x)

preds = Dense(1, activation='sigmoid')(embedded_sequences)

model = Sequential()

model.add(Dense(50, input_dim=MAX_SEQUENCE_LENGTH, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

tensorboard = TensorBoard(log_dir=os.path.join(ROOT_DIR, 'logs'), histogram_freq=1, batch_size=128, write_grads=True)

train_log_dir = os.path.join(ROOT_DIR, 'logs/train')
val_log_dir = os.path.join(ROOT_DIR, 'logs/validation')

train_tboard_logger = FilterTensorBoard(log_dir=train_log_dir, write_graph=False,
                                        write_images=False, log_regex=r'^(?!val).*')
val_tboard_logger = FilterTensorBoard(log_dir=val_log_dir, write_graph=False,
                                      write_images=False, log_regex=r"^val")
# model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc', recall])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_test, y_test),
          callbacks=[train_tboard_logger, val_tboard_logger])
