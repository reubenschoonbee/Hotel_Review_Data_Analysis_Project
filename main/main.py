from data_retrieval import update_data, retrieve_data, get_top_10, get_least_reliable, get_most_frequent_words
from plot_functions import plot_top_10, plot_least_reliable, plot_wc_score_relationship
from classification_model import words_classification_model_accuracy, word_count_classification_model_accuracy, create_negative_words, create_positive_words

# data retrieval and creation
data = retrieve_data()
negative_words = create_negative_words(data[:5000])
positive_words = create_positive_words(data[:5000])

# data analysis
top_10_hotels = get_top_10(data)
least_reliable_hotels = get_least_reliable(data)
most_frequent_words = get_most_frequent_words(data)

# data plotting
plot_top_10(top_10_hotels)
plot_least_reliable(least_reliable_hotels)
plot_wc_score_relationship(data)

# correlation between positive review wc and reviewer score
correlation = data['positive_review_wc'].corr(data['reviewer_score'])
print(f'Correlation between word count of positive review and reviewer score: {correlation}')

# classification model accuracy -
accuracy1 = words_classification_model_accuracy(data, negative_words, positive_words)
accuracy2 = word_count_classification_model_accuracy(data)
print(f'Positive and negative word correlatoin model accuracy: {accuracy1}')
print(f'Word count correlation model accuracy: {accuracy2}')
