from sklearn.linear_model import LogisticRegression

from src.custom_metrics import *
from src.helpers import *

(x_train, y_train), (x_test, y_test) = get_test_train_set()

model = LogisticRegression()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print('Test score:', score)

y = model.predict(x_test)

print(roc_auc_score(y_test, y))


# activations
# intermediate: relu, elu, selu, tanh, sigmoid
# last: softmax, sigmoid


#
# def text_fit(model, clf_model, coef_show=1):
#     tokenizer = get_tokenizer()
#     columns = getColumns(dataPath(), ROW_LIMIT)
#     texts, scores = columns
#     scores_array = np.reshape(np.asarray(scores), (-1, 1))
#
#     x_train_words, x_test_words, y_train, y_test = train_test_split(texts, scores_array, test_size=TEST_SPLIT,
#                                                                     random_state=RANDOM_SEED)
#
#     tokenizer.fit_on_texts(x_train_words)
#     x_train = tokenizer.texts_to_matrix(x_train_words, mode='tfidf')
#     x_test = tokenizer.texts_to_matrix(x_test_words, mode='tfidf')
#
#     print('# features: {}'.format(x_train.shape[1]))
#     print('# train records: {}'.format(x_train.shape[0]))
#     print('# test records: {}'.format(x_test.shape[0]))
#     clf = clf_model.fit(x_train, y_train)
#     acc = clf.score(x_test, y_test)
#     print('Model Accuracy: {}'.format(acc))
#
#     # if coef_show == 1:
#     #     w = model.get_feature_names()
#     #     coef = clf.coef_.tolist()[0]
#     #     coeff_df = pd.DataFrame({'Word': w, 'Coefficient': coef})
#     #     coeff_df = coeff_df.sort_values(['Coefficient', 'Word'], ascending=[0, 1])
#     #     print('')
#     #     print('-Top 20 positive-')
#     #     print(coeff_df.head(20).to_string(index=False))
#     #     print('')
#     #     print('-Top 20 negative-')
#     #     print(coeff_df.tail(20).to_string(index=False))
#
#
# # text_fit(X, y, c, LogisticRegression())
#
# tfidf = TfidfVectorizer(stop_words='english')
# text_fit(tfidf, LogisticRegression())
