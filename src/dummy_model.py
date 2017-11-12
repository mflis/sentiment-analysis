from sklearn.dummy import DummyClassifier

from src.custom_metrics import *
from src.helpers import *
from src.logger import *

tags = "random, full set"
print_source(__file__, tags)

(x_train, y_train), (x_test, y_test) = get_test_train_set(row_limit=1000)
model = DummyClassifier(random_state=RANDOM_SEED, strategy='most_frequent')

trained_classifier = model.fit(x_train, y_train)
accuracy = trained_classifier.score(x_test, y_test)

y = model.predict(x_test)
# show_confusion_matrix(y_test, y)
# plot_prec_recall(y_test, y, name_suffix='random')
# plot_roc_curve(y_test, y, name_suffix='random')
#
# y_pred = np.rint(y_pred_float).astype(int)
# auc = roc_auc_score(y_val, y_pred)
# precision, recall, f1, support = precision_recall_fscore_support(
#     y_val, y_pred, average='binary')
# kappa = cohen_kappa_score(y_val, y_pred)
# conf_matrix = confusion_matrix(y_val, y_pred)

print(roc_auc_score(y_test, y))
print(np.average(y))

print('Test accuracy: {}'.format(accuracy))
