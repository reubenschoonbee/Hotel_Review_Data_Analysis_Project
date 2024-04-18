import nltk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def create_negative_words(data):

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['negative_review'])
    X_array = X.toarray()

    feature_names = vectorizer.get_feature_names_out()

    correlations = [np.corrcoef(X_array[:, i], data['reviewer_score'])[0, 1] for i in range(X_array.shape[1])]
    correlation_df = pd.DataFrame({'word': feature_names, 'correlation': correlations})
    correlation_df = correlation_df.reindex(correlation_df.correlation.abs().sort_values(ascending=False).index)

    specific_words = correlation_df['word'].head(25).tolist()
    specific_words = [word for word in specific_words if word not in stop_words]

    with open('resources/negative_words.txt', 'w') as f:
        for word in specific_words:
            f.write(f'{word}\n')

    return specific_words

def create_positive_words(data):

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['positive_review'])
    X_array = X.toarray()

    feature_names = vectorizer.get_feature_names_out()

    correlations = [np.corrcoef(X_array[:, i], data['reviewer_score'])[0, 1] for i in range(X_array.shape[1])]
    correlation_df = pd.DataFrame({'word': feature_names, 'correlation': correlations})
    correlation_df = correlation_df.reindex(correlation_df.correlation.abs().sort_values(ascending=False).index)

    specific_words = correlation_df['word'].head(25).tolist()
    specific_words = [word for word in specific_words if word not in stop_words]

    with open('resources/positive_words.txt', 'w') as f:
        for word in specific_words:
            f.write(f'{word}\n')

    return specific_words

def remove_common_elements(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    list1_unique = list(set1 - set2)
    list2_unique = list(set2 - set1)

    return list1_unique, list2_unique

def words_classification_model_accuracy(data, negative_words, positive_words):

    remove_common_elements(negative_words, positive_words)
    for word in negative_words:
        data[f'contains_{word}'] = data['negative_review'].str.contains(word)
    for word in positive_words:
        data[f'contains_{word}'] = data['positive_review'].str.contains(word)

    data['high_score'] = data['reviewer_score'] > 9
    X = data[[f'contains_{word}' for word in negative_words + positive_words]]
    y = data['high_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_prediction)

    return accuracy

def word_count_classification_model_accuracy(data):

    data['high_score'] = data['reviewer_score'] > 9
    X = data[['negative_review_wc', 'positive_review_wc']]
    y = data['high_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_prediction)

    return accuracy


def calculate_sentiment_scores(data):
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    data['positive_review_sentiment'] = data['positive_review'].apply(lambda review: sia.polarity_scores(review)['compound'])
    data['negative_review_sentiment'] = data['negative_review'].apply(lambda review: sia.polarity_scores(review)['compound'])
    return data
