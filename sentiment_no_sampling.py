# Import all the necessary modules
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score

# User Defined Functions
def X_Vectorization(review_X, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, sublinear_tf=True, use_idf=True)
        transformed_review = vectorizer.fit_transform(review_X)
    else:
        transformed_review = vectorizer.transform(review_X)
    return transformed_review, vectorizer

def Classifier(X, y, to_predict, model=None):
    if model == 1:
        svm_classifier = svm.SVC(kernel='rbf')
        svm_classifier.fit(X, y)
        return svm_classifier.predict(to_predict)

    elif model == 2:
        randomforest_classifier = RandomForestClassifier(n_estimators=350, random_state=44)
        randomforest_classifier.fit(X, y)
        return randomforest_classifier.predict(to_predict)

    elif model == 3:
        gbm_classifier = GradientBoostingClassifier(n_estimators=550, learning_rate=0.2, max_depth=3, random_state=42)
        gbm_classifier.fit(X, y)
        return gbm_classifier.predict(to_predict)

def Model_Evaluation(true, pred, model_name):
    f1 = f1_score(true, pred, average='weighted')
    recall = recall_score(true, pred, average='weighted')
    print(f"F1 and Recall scores of {model_name}: ")
    print(f"F1 Score: {f1}")
    print(f"Recall Score: {recall}")
    return f1, recall

# Import Data
df = pd.read_csv('Dataset-SA.csv')
df.dropna(inplace=True)

summary_lst_X = df['Summary']
sentiments_lst_y = df['Sentiment']

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(summary_lst_X, sentiments_lst_y, test_size=0.25, random_state=53)

# Vectorization - Transform text data into vectors
train_vectors, vectorize = X_Vectorization(X_train)
test_vectors, _ = X_Vectorization(X_test, vectorize)

'''Classifiers'''
# To observe the F1 and Recall score of SVM model set "set_var=1"
# To observe the F1 and Recall score of Random Forest model set "set_var=2"
# To observe the F1 and Recall score of Gradient Boosting Machine model set "set_var=3"

set_var = 3  # Change this to 2 or 3 for different classifiers

if set_var == 1:
    svm_pred_result = Classifier(train_vectors, y_train, test_vectors, model=1)
    F1_Score, Recall_Score = Model_Evaluation(y_test, svm_pred_result, "SVM")

elif set_var == 2:
    rf_pred_result = Classifier(train_vectors, y_train, test_vectors, model=2)
    F1_Score, Recall_Score = Model_Evaluation(y_test, rf_pred_result, "Random Forest")

elif set_var == 3:
    gbm_pred_result = Classifier(train_vectors, y_train, test_vectors, model=3)
    F1_Score, Recall_Score = Model_Evaluation(y_test, gbm_pred_result, "Gradient Boosting Machine")
