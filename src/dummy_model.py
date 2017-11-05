from sklearn.dummy import DummyClassifier

from src.custom_metrics import *
from src.helpers import *

(x_train, y_train), (x_test, y_test) = get_test_train_set()
model = DummyClassifier(random_state=RANDOM_SEED)

trained_classifier = model.fit(x_train, y_train)
accuracy = trained_classifier.score(x_test, y_test)

y = model.predict(x_test)
show_confusion_matrix(y_test, y)
plot_prec_recall(y_test, y, name_suffix='random')
plot_roc_curve(y_test, y, name_suffix='random')
print(roc_auc_score(y_test, y))
print(np.average(y))

print('Test accuracy: {}'.format(accuracy))
