from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from src.custom_metrics import *
from src.helpers import *

(x_train, y_train), (x_test, y_test) = get_test_train_set()

model = Sequential()

model.add(Dense(int(VOCABULARY_LIMIT / 2), input_dim=VOCABULARY_LIMIT, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

auc = CustomMetrics(tags="softmax not learing")
history = model.fit(x_train, y_train, batch_size=4096, epochs=10,
                    validation_data=(x_test, y_test), callbacks=[auc])

score = model.evaluate(x_test, y_test, batch_size=4096, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y = model.predict(x_test)

print(roc_auc_score(y_test, y))
print(np.average(y))



# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid
