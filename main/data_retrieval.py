import pandas as pd
import nltk
from nltk.corpus import stopwords

# writes additional columns back to the csv file
def update_data():
    data = pd.read_csv('resources\hotels.csv')
    data['positive_review_wc'] = data['positive_review'].apply(lambda x: len(x.split()))
    data['negative_review_wc'] = data['negative_review'].apply(lambda x: len(x.split()))
    data.to_csv('resources\hotels.csv', index=False)

# retrieves the data from the csv file
def retrieve_data():
    data = pd.read_csv('resources\hotels.csv')
    return data

#  returns the top 10 hotels with the most reviews
def get_top_10(data):
    reviews_per_hotel = data['hotel_name'].value_counts()
    top_10_hotels = reviews_per_hotel.head(10)
    return top_10_hotels

# helper function for getting interquartile range
def inter_quartile_range(x):
    return x.quantile(0.75) - x.quantile(0.25)

# returns the 10 least reliable hotels based on the interquartile range of reviewer scores
def get_least_reliable(data):
    iqr_review_ratings = data.groupby('hotel_name')['reviewer_score'].agg(inter_quartile_range)
    least_reliable_hotels = iqr_review_ratings.sort_values(ascending=False).head(10)
    return least_reliable_hotels

# returns the 10 most frequent words in the positive reviews

def get_most_frequent_words(data):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    most_frequent_words = data['positive_review'].str.lower().str.split(expand=True).stack()
    most_frequent_words = most_frequent_words[~most_frequent_words.isin(stop_words)]
    most_frequent_words = most_frequent_words.value_counts().head(10)
    return most_frequent_words
