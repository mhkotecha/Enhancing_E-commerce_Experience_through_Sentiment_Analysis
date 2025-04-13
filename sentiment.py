#Import all the necessary modules
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score

import webbrowser
import urllib.parse
import time

#User Defined Functions
def X_Vecorization(review_X, vectorizer=None):

    if vectorizer is None:
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.75, sublinear_tf=True, use_idf=True)
        transformed_review = vectorizer.fit_transform(review_X)
    else:
        transformed_review = vectorizer.transform(review_X)

    return transformed_review, vectorizer

def Sampling(trainvectors,ytrain,method=None):

    # Random Under-Sampling
    if method==1:
        sampling_strategy = {'positive': len(ytrain[ytrain == 'negative'])}  # Under-sample 'positive' to match 'negative' count
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
        Xresampled, yresampled = rus.fit_resample(trainvectors, ytrain)
        return Xresampled, yresampled, "Random Under-Sampling: "

    # Tomek Links - Under Sampling Technique
    elif method==2:
        tomek_links = TomekLinks(sampling_strategy='auto')
        Xresampled, yresampled = tomek_links.fit_resample(trainvectors, ytrain)
        return Xresampled, yresampled, "Tomek Links Under-sampling: "

    # SMOTE - Over Sampling technique + Random Under-Sampling
    elif method==3:
        sampling_strategy = {'positive': len(ytrain[ytrain == 'negative'])}  # Under-sample 'positive' to match 'negative' count
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
        Xresampled, yresampled = rus.fit_resample(trainvectors, ytrain)
        smote = SMOTE()
        Xresampled, yresampled = smote.fit_resample(Xresampled,yresampled)  # Over-sample 'neutral' to match 'negative' count
        return Xresampled, yresampled, "SMOTE Over-sampling + Random Under-Sampling: "

def Classifier(X,y,to_predict,model=None):
    if model==1:
        svm_classifier = svm.SVC(kernel='rbf')
        svm_classifier.fit(X, y)
        svm_prediction = svm_classifier.predict(to_predict)
        return svm_prediction
    elif model==2:
        randomforest_classifier = RandomForestClassifier(n_estimators=350, random_state=44)
        randomforest_classifier.fit(X, y)
        randomforest_prediction = randomforest_classifier.predict(to_predict)
        return randomforest_prediction
    elif model==3:
        gbm_classifier = GradientBoostingClassifier(n_estimators=550, learning_rate=0.2, max_depth=3, random_state=42)
        gbm_classifier.fit(X, y)
        gbm_prediction = gbm_classifier.predict(to_predict)
        return gbm_prediction

def Model_Evaluation(true,pred,modl):
    f1 = f1_score(true,pred,average='weighted')
    recall = recall_score(true,pred,average='weighted')
    print("F1 and Recall scores of "+ modl+": ")
    print("F1 Score: ", f1)
    print("Recall Score: ", recall)
    return f1,recall

def open_email_client():
    to_email = "komalb023269@gmail.com"
    subject = "Issue with the product"

    # Construct the Gmail compose URL
    gmail_url = f"https://mail.google.com/mail/?view=cm&to={to_email}&su={urllib.parse.quote(subject)}"

    webbrowser.open_new_tab(gmail_url)


''' Import Data'''
#Import data
df = pd.read_csv('/Users/komalb/Downloads/Dataset-SA.csv')
#Drop rows that contains null values
df.dropna(inplace=True)

summary_lst_X = df['Summary']
sentiments_lst_y = df['Sentiment']

''' Split train and test data'''
X_train, X_test, y_train, y_test = train_test_split(summary_lst_X,sentiments_lst_y,test_size=0.25, random_state=53)

''' Vectorization - Transform text data into vectors'''
train_vectors, vectorize = X_Vecorization(X_train)
test_vectors, _ = X_Vecorization(X_test, vectorize)

''' Various Sampling Techniques'''
#Data is imbalance, so we need to perform sampling
X_resampled, y_resampled,technique = Sampling(train_vectors, y_train, method=1)

''' Data Visualization'''
before_sampling = Counter(sentiments_lst_y)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Sentiment Distribution Before and After Sampling')

# Plotting the initial distribution
axes[0, 0].bar(before_sampling.keys(), before_sampling.values(), color='#e6655a')
axes[0, 0].set_title('Initial Sentiment Distribution')
axes[0, 0].set_xlabel('Sentiment')
axes[0, 0].set_ylabel('Count')
for key, value in before_sampling.items():
    axes[0, 0].text(key, value, str(value), ha='center', va='bottom')

# Function to plot the rest of the distributions
def plot_distribution(ax, method, title,colour):
    X_resampled, y_resampled, _ = Sampling(train_vectors, y_train, method)
    after_sampling = Counter(y_resampled)
    ax.bar(after_sampling.keys(), after_sampling.values(), color=colour)
    ax.set_title(title)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    for key, value in after_sampling.items():
        ax.text(key, value, str(value), ha='center', va='bottom')

# Plotting for each sampling method
plot_distribution(axes[0, 1], 1, 'Sentiment Distribution after RUS',colour='#34c9eb')
plot_distribution(axes[1, 0], 2, 'Sentiment Distribution after Tomek Links',colour='#3cc977')
plot_distribution(axes[1, 1], 3, 'Sentiment Distribution after RUS + SMOTE',colour='#b8bd39')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

'''Classifiers'''
#To observe the F1 and Recall score of SVM model set "set_var=0"
#To observe the F1 and Recall score of Random Forest model set "set_var=1"
#To observe the F1 and Recall score of Gradient Boosting Machine model set "set_var=3"

set_var = 0

if set_var==1:
    svm_pred_result = Classifier(X_resampled, y_resampled, test_vectors, model=1)
    F1_Score, Recall_Score = Model_Evaluation(y_test, svm_pred_result,"SVM") #Evaluate the SVM model

elif set_var==2:
    rf_pred_result = Classifier(X_resampled, y_resampled, test_vectors, model=2)
    F1_Score, Recall_Score = Model_Evaluation(y_test, rf_pred_result, "Random Forest") #Evaluate the RF model

elif set_var == 3:
    gbm_pred_result = Classifier(X_resampled, y_resampled, test_vectors, model=3)
    F1_Score, Recall_Score = Model_Evaluation(y_test, gbm_pred_result, "Gradient Boosting Machine")  # Evaluate the GBM model


#Custom Instructions section
print("Hi, I am your sentimental analysis model. Please write your review")
user_review = input("Enter Review: ")
print("Which model do you want to analyze your review?")
print("1: SVM      2: Random Forest    3. Gradient Boosting Classifier")
choosen_model = int(input("Enter the option: "))
if choosen_model<0 or choosen_model>3:
    print("You entered invalid option")
    exit()

print("Please wait while we analyze your review...")
user_review_vectorized, _ = X_Vecorization([user_review], vectorizer=vectorize)
res = Classifier(X_resampled, y_resampled, user_review_vectorized, model=choosen_model)
print(res)

if "negative" in res:
    print("We sincerely apologise for the inconvenience. Please take a moment to explain the issue, so that we can resolve it as soon as possible.")
    print("Redirecting you to email service")
    time.sleep(5)
    open_email_client()

elif "neutral" in res:
    print("Looks like you are not completely satisfied with the product. Would you mind answering the survey to help us improve.")
    time.sleep(3)
    webbrowser.open('https://docs.google.com/forms/d/e/1FAIpQLScRWypGHynaUn_oVnxpyJ8w-PYZKZ-gYlsO7H6Vw1vSwMsvpQ/viewform')

elif "positive" in res:
    print("Glad to hear that you liked our product. Thank you for shopping with us")