from tensorflow.contrib.keras.python.keras.layers import Dense
from tensorflow.contrib.keras.python.keras.models import Sequential

from src.auc_callback import *
from src.helpers import *

(x_train, y_train), (x_test, y_test) = get_test_train_set()

model = Sequential()
model.add(Dense(2500, activation='sigmoid', input_dim=VOCABULARY_LIMIT))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

auc = AucMetric((x_test, y_test))
history = model.fit(x_train, y_train, batch_size=512, epochs=2,
                    validation_data=(x_test, y_test), callbacks=[auc])

model.fit(x_train, y_train)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y = model.predict(x_test)

print(roc_auc_score(y_test, y))



# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid
