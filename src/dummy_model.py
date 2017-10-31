from sklearn.dummy import DummyClassifier

from src.helpers import *

(x_train, y_train), (x_test, y_test) = get_test_train_set()
model = DummyClassifier(random_state=RANDOM_SEED)

trained_classifier = model.fit(x_train, y_train)
accuracy = trained_classifier.score(x_test, y_test)

print('Test accuracy: {}'.format(accuracy))
