import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

def clean_text(text):
    """
    Clean the text by converting it to lowercase, removing punctuation, and removing numbers.

    Parameters:
    text (str): The text to be processed.

    Returns:
    str: The cleaned text.
    """
    text = text.lower() 
    text = re.sub(r'[^\w\s]', '', text) 
    text = re.sub(r'\d+', '', text) 
    return text

def tokenize_text(text):
    """
    Tokenize the text by splitting it into words based on whitespace.

    Parameters:
    text (str): The cleaned text.

    Returns:
    list: A list of words from the text.
    """
    return text.split()

def remove_stopwords(word_list):
    """
    Remove stopwords from the list of words.

    Parameters:
    word_list (list): A list of words.

    Returns:
    list: A list of words with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in word_list if word not in stop_words]

def stem_text(word_list):
    """
    Apply stemming to a list of words. Stemming is the process of reducing words to their word stem, base, or root form.
    For example, words like "running", "runs", "ran" are all reduced to the base form "run".

    Parameters:
    word_list (list of str): A list of words to be stemmed.

    Returns:
    list of str: A list of stemmed words.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in word_list]

def recombine_text(word_list):
    """
    Recombine the list of words into a single string of text.

    Parameters:
    word_list (list): A list of words.

    Returns:
    str: A single string formed by combining the words.
    """
    return ' '.join(word_list)

def text_preprocessing(text):
    """
    Apply a series of text preprocessing steps to a given text.
    Steps include:
    1. Cleaning the text by converting it to lowercase, removing punctuation, and numbers.
    2. Tokenizing the cleaned text into words.
    3. Removing English stopwords from the tokenized words.
    4. Applying stemming to reduce words to their root form.
    5. Recombining the stemmed words back into a single string.
    
    Parameters:
    text (str): The text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    cleaned = clean_text(text)
    tokenized = tokenize_text(cleaned)
    filtered = remove_stopwords(tokenized)
    lemmatized = stem_text(filtered)
    return recombine_text(lemmatized)

def preprocess_data(data):
    """
    Process a list of email data. Each email is represented as a dictionary with keys 'subject', 'data', and 'ham/spam'.
    The function applies text preprocessing to 'subject' and 'data' fields and converts 'ham/spam' into a binary label (1 for spam, 0 for ham).
    
    Parameters:
    data (list of dict): The list of email data, where each email is a dictionary.

    Returns:
    list of dict: The list of processed email data, with each email having preprocessed 'subject', 'data', and a binary 'ham/spam' label.
    """
    processed_data = []

    for item in data:
        processed_subject = text_preprocessing(item['subject']) if item['subject'] else ''
        processed_body = text_preprocessing(item['data'])
        spam_ham = 1 if item['ham/spam'] == 'spam' else 0

        processed_data.append({
            'subject': processed_subject,
            'data': processed_body,
            'ham/spam': spam_ham
        })

    return processed_data

