import matplotlib.pyplot as plt

def plot_top_10(top_10_hotels):
    top_10_hotels.plot(kind='bar')
    plt.title('Top 10 Hotels with the Most Reviews')
    plt.xlabel('Hotel Name', fontsize=6)
    plt.ylabel('Number of Reviews')
    plt.xticks(fontsize=8)
    plt.subplots_adjust(bottom=0.4)
    plt.show()

def plot_least_reliable(least_reliable_hotels):
    least_reliable_hotels.plot(kind='bar')
    plt.title('10 Least Reliable Hotels')
    plt.title('10 Least Reliable Hotels')
    plt.xlabel('Hotel Name', fontsize=6)
    plt.ylabel('Interquartile Range of Reviewer Scores')
    plt.xticks(fontsize=8)
    plt.subplots_adjust(bottom=0.4)
    plt.show()

def plot_wc_score_relationship(data):
    plt.figure(figsize=(18, 13))
    plt.scatter(data['positive_review_wc'], data['reviewer_score'])
    plt.title('Word Count of Positive Review vs. Reviewer Score')
    plt.xlabel('Word Count of Positive Review', fontsize=20)
    plt.ylabel('Reviewer Score', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=15)  # Increase tick label size
    plt.show()