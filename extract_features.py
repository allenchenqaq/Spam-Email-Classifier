import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(input_json, output_csv, max_features_subject, max_features_body):
    """
    Extracts TF-IDF features from both the subject and body of emails.

    Parameters:
    input_json (str): Path to the JSON file containing processed email data.
    output_csv (str): Path to save the CSV file with extracted features.
    max_features_subject (int): Number of most important features to extract from the subject.
    max_features_body (int): Number of most important features to extract from the body.

    Returns:
    None: This function saves the extracted features to a CSV file.
    """

    # Load email data
    with open(input_json, 'r') as file:
        email_data = json.load(file)

    # Extract subject, body, and labels
    subjects = [email['subject'] for email in email_data]
    bodies = [email['data'] for email in email_data]
    labels = [email['ham/spam'] for email in email_data]

    # Compute TF-IDF for subject
    vectorizer_subject = TfidfVectorizer(max_features=max_features_subject)
    tfidf_matrix_subject = vectorizer_subject.fit_transform(subjects)

    # Compute TF-IDF for body
    vectorizer_body = TfidfVectorizer(max_features=max_features_body)
    tfidf_matrix_body = vectorizer_body.fit_transform(bodies)

    # Convert to DataFrame
    feature_names_subject = vectorizer_subject.get_feature_names_out()
    feature_names_body = vectorizer_body.get_feature_names_out()
    email_features_subject = pd.DataFrame(tfidf_matrix_subject.toarray(), columns=feature_names_subject)
    email_features_body = pd.DataFrame(tfidf_matrix_body.toarray(), columns=feature_names_body)

    # Concatenate subject and body features
    email_features = pd.concat([email_features_subject, email_features_body], axis=1)

    # Add ham/spam column
    email_features['ham/spam'] = labels

    # Save to CSV
    email_features.to_csv(output_csv, index=False)