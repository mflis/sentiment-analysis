from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.utils import plot_model
from tensorflow.python.layers.core import Dense

model = Sequential()
model.add(Dense(10, input_shape=((20,))))
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',  # as in paper, article uses adam
              metrics=['acc'])

plot_model(model, to_file='model.png')
