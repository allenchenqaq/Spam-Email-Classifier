# Spam Email Classifier

## Overview
This project is a sophisticated spam email classifier utilizing the extensive Enron Email Dataset, which comprises over 500,000 emails. Our goal is to accurately distinguish spam emails from regular ones using advanced data preprocessing and machine learning techniques.

## Data Preparation
We start by merging data from multiple compressed files into a single CSV file, followed by basic data cleaning. This sets the stage for the more intricate preprocessing phase.

## Preprocessing
The preprocessing phase involves:
- Removing punctuation and numbers
- Converting text to lowercase
- Tokenizing text into words
- Removing stop words
- Applying stemming to reduce words to their root form
- Recombining the stemmed words back into a single string

## Feature Extraction
We employ the TF-IDF (Term Frequency-Inverse Document Frequency) method to extract features from both the subject and body of emails. This technique helps in evaluating the importance of a word in a document relative to a collection of documents.

- **Subject**: Extraction of the top 20 TF-IDF features
- **Body**: Extraction of the top 1000 TF-IDF features  

These features are selected based on their TF-IDF scores, ensuring that the most relevant features for classification are utilized.

## Data Analysis
We employ various statistical tests to determine whether there is a significant difference in word frequencies between spam and ham emails, including:
- Normality test
- Levene's test
- Mann-Whitney U test
- Chi-squared test

## Models
We employ various models for training and validation and compare their results, including:
- Naive Bayes Classifier
- Random Forest Classifier
- Logistic Regression Classifier
- Support Vector Classifier (SVC)

## Usage
To install the necessary dependencies, run: `pip install -r requirements.txt`.  
After installation, execute the program by running `main.py`. Note that the process is somewhat time-intensive and may take approximately 5 minutes.

## Requirements
- pandas
- numpy
- ntlk
- sklearn
- matplotlib
- seaborn

## Contributing
This project has been a collaborative effort by:
- John Wang
- Allen Chen
- Shiyuan Miao

