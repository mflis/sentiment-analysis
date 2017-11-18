from sklearn.dummy import DummyClassifier

from src.mechanics.postprocessing.custom_metrics import *
from src.mechanics.postprocessing.logger import *
from src.mechanics.postprocessing.metric_plots import plot_prec_recall, plot_roc_curve
from src.mechanics.preprocessing.helpers import *

tags = "random, full set"
print_source(__file__, tags)

(x_train, y_train), (x_test, y_test) = get_test_train_set(row_limit=20000)
model = DummyClassifier(random_state=RANDOM_SEED, strategy='most_frequent')

trained_classifier = model.fit(x_train, y_train)
accuracy = trained_classifier.score(x_test, y_test)

y_pred_float = model.predict(x_test)
show_confusion_matrix(y_test, y_pred_float)

y_pred = np.rint(y_pred_float).astype(int)
auc = roc_auc_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average='binary')
kappa = cohen_kappa_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
plot_prec_recall(y_test, y_pred, name_suffix='random')
plot_roc_curve(y_test, y_pred, name_suffix='random')
print("ROC AUC score: {}".format(auc))
