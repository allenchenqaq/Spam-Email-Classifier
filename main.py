import json
import os
import pandas as pd
from get_raw_data import main as get_raw_data_main
from preprocess_data import preprocess_data
from extract_features import extract_features
from train_and_validate_model import naive_model, random_forest_model, logistic_regression_model, SVC_model
from dictionary_generator import create_dictionary_on_all_email
from scipy import stats
from plot_model_performance import plot_model_performance
import matplotlib.pyplot as plt
import seaborn as sns

MAX_FEATURES_SUBJECT = 20
MAX_FEATURES_BODY = 1000


def main():
    # Extract and process raw data from tar files
    if not os.path.exists('output/enron_email_data.json'):
        print('getting raw data...')
        get_raw_data_main()

        
    # Process the raw data (cleaning, tokenizing, removing stopwords, stemming, etc.)
    if not os.path.exists('output/processed_email_data.json'):
        print('preprocessing raw data...')
        with open('output/enron_email_data.json', 'r') as file:
            raw_data = json.load(file)
        processed_data = preprocess_data(raw_data)

        # Save the processed data to a new JSON file
        print('saving processed data...')
        with open('output/processed_email_data.json', 'w', encoding='utf-8') as file:
            json.dump(processed_data, file, ensure_ascii=False, indent=4)

            
    # Create dictionary(the occurence of each word in all spam/ham email) based on processed_data,
    if not os.path.exists('output/processed_email_dictionary.json'):
        print('creating dictionary...')
        with open('output/processed_email_data.json', 'r') as file:
            processed_data = json.load(file)
        dictionary = create_dictionary_on_all_email(processed_data)

        print('Saving dictionary...')
        dictionary.to_csv('output/processed_email_dictionary.csv', index=False)

        
    # which words are most indicative of spam emails? Which ones are prevalent in legitimate emails?
    if os.path.exists('output/processed_email_dictionary.csv'):
        data = pd.read_csv('output/processed_email_dictionary.csv')

        spam_data = data[data['ham/spam'] == 1]
        ham_data = data[data['ham/spam'] == 0]
        
        
        # is there a significant difference in word frequencies between spam and ham emails?
        # test normality and equal variance
        spam_freq = spam_data['count']
        ham_freq = ham_data['count']
        _, spam_normality = stats.normaltest(spam_freq)
        _, ham_normality = stats.normaltest(ham_freq)
        _, variance_equal = stats.levene(spam_freq, ham_freq)
        print(f"Normality test - Spam data: p-value = {spam_normality}")
        print(f"Normality test - Ham data: p-value = {ham_normality}")
        print(f"Equal Variance test: p-value = {variance_equal}")

        # apply a mannwhitneyu test
        _, p_value = stats.mannwhitneyu(spam_freq, ham_freq)
        print(f"U-statistic P-value: {p_value}")
        if p_value < 0.05:
            print("There's a significant difference in word frequencies between spam and ham emails.")
        else:
            print("No significant difference found in word frequencies between spam and ham emails.")

        # is there arelationship between word existence and spam/ham
        contingency_table = pd.crosstab(data['exist'], data['ham/spam'])
        # Apply the chi-square test
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)

        print(f"Chi-square statistic: {chi2}")
        print(f"P-value: {p_value}")

        # Check for significance
        if p_value < 0.05:
            print("There's a significant difference in the existence of words between spam and ham emails.")
        else:
            print("No significant difference found in the existence of words between spam and ham emails.")

        
        # print the top indicative words in spam and ham
        # filter data where 'exist' column is equal to 1
        spam_data = spam_data[spam_data['exist'] == 1]
        ham_data = ham_data[ham_data['exist'] == 1]

        # calculate total counts for spam and ham emails
        total_spam = spam_data['count'].sum()
        total_ham = ham_data['count'].sum()

        # calculate relative frequencies of words in spam and ham emails
        spam_data = spam_data.copy()
        ham_data = ham_data.copy()
        spam_data['spam_freq'] = spam_data['count'] / total_spam
        ham_data['ham_freq'] = ham_data['count'] / total_ham

        # calculate the ratio of spam to ham frequencies for each word
        data_merged = spam_data.merge(ham_data, on='words', how='outer')
        data_merged['spam_to_ham_ratio'] = data_merged['spam_freq'] / (data_merged['ham_freq'].replace(0, 0.0001))  # to avoid division by zero

        # sort the data by spam-to-ham ratio to identify indicative words
        indicative_spam_words = data_merged.sort_values(by='spam_to_ham_ratio', ascending=False).head(10)
        prevalent_ham_words = data_merged.sort_values(by='spam_to_ham_ratio').head(10)

        indicative_spam_words['count'] = indicative_spam_words['count_x'].astype(int)
        prevalent_ham_words['count'] = prevalent_ham_words['count_y'].astype(int)
        # print('all spam: ')
        # print(indicative_spam_words)
        # print('all ham: ')
        # print(prevalent_ham_words)
        
        # print the indicative words
        print("Top 10 indicative words for spam:")
        print(indicative_spam_words[['words', 'count', 'spam_to_ham_ratio']])

        print("\nTop 10 prevalent words in ham:")
        print(prevalent_ham_words[['words', 'count','spam_to_ham_ratio']])   
        

        # Visualize Indicative Words
        plt.figure(figsize=(12, 6))
        sns.barplot(x='words', y='spam_to_ham_ratio', data=indicative_spam_words)
        plt.title('Top 10 Indicative Words for Spam')
        plt.xlabel('Words')
        plt.ylabel('Spam-to-Ham Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join('output', 'indicative_words_spam.png'))
        # plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='words', y='spam_to_ham_ratio', data=prevalent_ham_words)
        plt.title('Top 10 Prevalent Words in Ham')
        plt.xlabel('Words')
        plt.ylabel('Spam-to-Ham Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join('output', 'prevalent_words_ham.png'))
        # plt.show()
        
    # Extract features from the processed data
    if os.path.exists('output/email_features.csv'):
        print('extracting features...')
        extract_features('output/processed_email_data.json', 'output/email_features.csv', MAX_FEATURES_SUBJECT, MAX_FEATURES_BODY)

    # Train and validate the model
    print('training and validating model...\n')

    model_metrics = []

    # train naive model to determine spam/ham
    print('Training NAIVE BAYES CLASSIFIER...\n')
    _, acc, prec, rec, f1 = naive_model('output/email_features.csv')
    model_metrics.append(('Naive Bayes', acc, prec, rec, f1))

    # train random forest model to determine spam/ham
    print('Training RANDOM FOREST CLASSIFIER...\n')
    _, acc, prec, rec, f1 = random_forest_model('output/email_features.csv')
    model_metrics.append(('Random Forest', acc, prec, rec, f1))

    # train logistic regression model to determine spam/ham
    print('Training LOGISTIC REGRESSION CLASSIFIER...\n')
    _, acc, prec, rec, f1 = logistic_regression_model('output/email_features.csv')
    model_metrics.append(('Logistic Regression', acc, prec, rec, f1))

    # train SVC model to determine spam/ham
    print('Training SVC...\n')
    _, acc, prec, rec, f1 = SVC_model('output/email_features.csv')
    model_metrics.append(('SVC', acc, prec, rec, f1))

    plot_model_performance(model_metrics)

    print('done.')
if __name__ == "__main__":
    main()
