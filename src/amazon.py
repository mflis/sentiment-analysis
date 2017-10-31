from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential

from src.helpers import *

(x_train, y_train), (x_test, y_test) = get_test_train_set()


model = Sequential()
model.add(Dense(1, activation='softmax', input_dim=VOCABULARY_LIMIT))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=4096, epochs=10, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('../model.h5')


# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid
