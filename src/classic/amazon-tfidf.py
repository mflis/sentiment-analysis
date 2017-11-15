from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.regularizers import l2

from src.mechanics.postprocessing.custom_metrics import *
from src.mechanics.postprocessing.logger import *
from src.mechanics.postprocessing.plots import *
from src.mechanics.preprocessing.helpers import *

tags = "l2=0.01 10k words"
print_source(__file__, tags)

(x_train, y_train), (x_test, y_test) = get_test_train_set(row_limit=60000, undersample=True)

model = Sequential()

model.add(Dense(VOCABULARY_LIMIT, input_dim=VOCABULARY_LIMIT, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

custom_metrics = CustomMetrics(tags)
csv = CSVLogger('../csv_logs/{}-{}.csv'.format(src.CURRENT_TIME, tags))
history = model.fit(x_train, y_train, batch_size=4096, epochs=15,
                    validation_data=(x_test, y_test), callbacks=[custom_metrics, csv])
plot_metric(history, 'auc')

score = model.evaluate(x_test, y_test, batch_size=4096, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y = model.predict(x_test)

print(roc_auc_score(y_test, y))
print(np.average(y))

# todo  read about statistical significance
# todo to read: http://cs231n.github.io/neural-networks-3/
# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid


# todo : to try
# use top 10 k words - https://www.kaggle.com/ruzerichards/predicting-amazon-reviews-using-cnns
