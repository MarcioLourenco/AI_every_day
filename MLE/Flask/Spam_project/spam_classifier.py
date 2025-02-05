import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score


def spam_detector(train_df, valid_df, test_df):
    vectorizer = TfidfVectorizer(
        min_df=2,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df['text'])

    with open(".\\data\\tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    X_valid = vectorizer.transform(valid_df['text'])
    X_test = vectorizer.transform(test_df['text'])

    logistic_model = LogisticRegression(
        random_state=0
    )
    logistic_model.fit(X_train, train_df['label'])

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train, train_df['label'])

    decision_tree_model = DecisionTreeClassifier(
        random_state=0
    )
    decision_tree_model.fit(X_train, train_df['label'])

    svc_model = LinearSVC()
    svc_model.fit(X_train, train_df['label'])

    y_pred_lm = logistic_model.predict(X_valid)
    y_pred_nb = naive_bayes_model.predict(X_valid)
    y_pred_dt = decision_tree_model.predict(X_valid)
    y_pred_svc = svc_model.predict(X_valid)

    models = [logistic_model, naive_bayes_model, decision_tree_model, svc_model]

    best_model_name, best_model = get_best_model(models, X_valid ,valid_df['label'])

    results = {
        "LogisticRegression": confusion_matrix(valid_df['label'], y_pred_lm),
        "MultinomialNB": confusion_matrix(valid_df['label'], y_pred_nb),
        "DecisionTreeClassifier": confusion_matrix(valid_df['label'], y_pred_dt),
        "LinearSVC": confusion_matrix(valid_df['label'], y_pred_svc),
        "BestClassifier": best_model_name,
        "TfidfVectorizer": X_test,
        "Prediction": best_model.predict(X_test),
        "Model": best_model
    }

    return results

def get_best_model(models, X_val, y_val):
    best_model = None
    best_f1_score = 0
    best_model_name = ""

    for model in models:
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        print(f"model: {model}, f1: {f1}")
        if f1 > best_f1_score:
            best_model = model
            best_f1_score = f1
            best_model_name = model.__class__.__name__

    return best_model_name, best_model

