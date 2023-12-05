import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words

def create_dictionary_on_all_email(processed_data):
    """
    Process a list of email data. Each email is represented as a dictionary with keys 'subject', 'data', and 'ham/spam'.
    The function applies text preprocessing to 'subject' and 'data' fields.

    Parameters:
    processed_data (list of dict): The list of email data, where each email is a dictionary with keys 'subject', 'data', and 'ham/spam'.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing word counts for all emails combined. Columns include 'word', 'spam/ham', 'count','type', and 'exist'.
    """
    # Create a DataFrame from the processed data
    df = pd.DataFrame(processed_data)

    # Combine subject and data into a single list of words
    df['words'] = df['subject'] + ' ' + df['data']
    df['words'] = df['words'].apply(lambda x: x.split())

    # Explode the list of words into separate rows
    df_exploded = df.explode('words')

    # Count the occurrences of each word
    word_counts = df_exploded.groupby(['words', 'ham/spam']).size().reset_index(name='count')

    # Create a 'type' column indicating whether the word is from the subject or data
    word_counts['type'] = np.where(word_counts['words'].isin(df['subject']), 'subject', 'data')

    # Create an 'exist' column indicating whether each word is in the NLTK words
    nltk_words_set = set(words.words())
    word_counts['exist'] = word_counts['words'].apply(lambda x: 1 if x in nltk_words_set else 0)

    return word_counts

