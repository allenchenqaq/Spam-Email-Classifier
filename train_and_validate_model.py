import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import uniform

def naive_model(features_csv, test_size=0.2, random_state=42):
    """
    Trains and validates a Naive Bayes classifier on the given email features.

    Parameters:
    features_csv (str): Path to the CSV file containing email features and labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    None: This function prints the accuracy and classification report of the model.
    """

    # Load data
    data = pd.read_csv(features_csv)
    labels = data['ham/spam']
    features = data.drop('ham/spam', axis=1)

    # Split dataset
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Initialize model
    model = MultinomialNB()

    # Train model
    model.fit(X_train, y_train)

    # Validate model
    predictions = model.predict(X_valid)

    # Print validation scores
    accuracy = accuracy_score(y_valid, predictions)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_valid, predictions))

    # Get model performance
    precision = precision_score(y_valid, predictions)
    recall = recall_score(y_valid, predictions)
    f1 = f1_score(y_valid, predictions)
    
    return model, accuracy, precision, recall, f1

def random_forest_model(features_csv, test_size=0.2, random_state=42):
    """
    Trains and validates a Random Forest classifier on the given email features.

    Parameters:
    features_csv (str): Path to the CSV file containing email features and labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    None: This function prints the accuracy and classification report of the model.
    """

    # Load data
    data = pd.read_csv(features_csv)
    labels = data['ham/spam']
    features = data.drop('ham/spam', axis=1)

    # Split dataset
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # used randomsearchcv to find the best params to use for random forest classifier
    # Best Hyperparameters: {'bootstrap': False, 'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 122}
    # 
    # 
    # # Define the parameter distributions to sample from
    # param_grid = {
    #     'n_estimators': randint(50, 200),
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap':[True, False]
    # }
    # # Initialize Random Forest classifier
    # rf_classifier = RandomForestClassifier()
    # # Initialize RandomizedSearchCV
    # random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_grid, cv = 10, verbose=2, n_jobs = 4)

    # Initialize model
    model = RandomForestClassifier(n_estimators=122, max_depth=30, min_samples_split=2, min_samples_leaf=2, max_features='sqrt', bootstrap=False)

    # Train model
    model.fit(X_train, y_train)

    # Validate model
    predictions = model.predict(X_valid)

    # Print validation scores
    accuracy = accuracy_score(y_valid, predictions)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_valid, predictions))
    
    # Get model performance
    precision = precision_score(y_valid, predictions)
    recall = recall_score(y_valid, predictions)
    f1 = f1_score(y_valid, predictions)

    return model, accuracy, precision, recall, f1

def logistic_regression_model(features_csv, test_size=0.2, random_state=42):
    """
    Trains and validates a Logistic Regression classifier with hyperparameter tuning using RandomizedSearchCV.

    Parameters:
    features_csv (str): Path to the CSV file containing email features and labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    None: This function prints the accuracy and classification report of the model.
    """

    # Load data
    data = pd.read_csv(features_csv)
    labels = data['ham/spam']
    features = data.drop('ham/spam', axis=1)

    # Split dataset
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    
    # Initialize model, standard scaler, and Principal Component Analysis
    model = LogisticRegression(penalty="l2", C=83.24526408004218)
    std_slc = StandardScaler()
    pca = PCA(n_components=386)
    # Define pipeline for GridSearchCV
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('logistic_Reg', model)])

    # used randomsearchcv to find the best params to use for logistic regression
    # Best Hyperparameters: {'logistic_Reg__C': 83.24526408004218, 'logistic_Reg__penalty': 'l2', 'pca__n_components': 386}
    # # Define hyperparameter distributions
    # parameters = {
    #     'pca__n_components': list(range(1, 2000, 1)),
    #     'logistic_Reg__C': uniform(0.001, 100),  # Use uniform distribution for C
    #     'logistic_Reg__penalty': ['l1', 'l2']
    # }
    # # Initialize RandomizedSearchCV
    # random_search = RandomizedSearchCV(pipe, parameters, n_iter=10, cv=5, scoring='accuracy', random_state=random_state)
    # Print the best hyperparameters
    # print("Best Hyperparameters:", random_search.best_params_)

    # Train model
    pipe.fit(X_train, y_train)

    # Validate model
    predictions = pipe.predict(X_valid)

    # Print validation scores
    accuracy = accuracy_score(y_valid, predictions)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_valid, predictions))

    # Get model performance
    precision = precision_score(y_valid, predictions)
    recall = recall_score(y_valid, predictions)
    f1 = f1_score(y_valid, predictions)
    
    return pipe, accuracy, precision, recall, f1

def SVC_model(features_csv, test_size=0.2, random_state=42):
    """
    Trains and validates a Support Vector Classifier (SVC) model with hyperparameter tuning using RandomizedSearchCV.    

    Parameters:
    features_csv (str): Path to the CSV file containing email features and labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
    None: This function prints the accuracy and classification report of the model.
    """

    # Load data
    data = pd.read_csv(features_csv)
    labels = data['ham/spam']
    features = data.drop('ham/spam', axis=1)

    # Split dataset
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    
    # Initialize model, standard scaler, and Principal Component Analysis
    model = SVC(C=9.799098521619943, gamma='auto', kernel='rbf')
    std_slc = StandardScaler()
    pca = PCA(n_components=309)
    # Define pipeline for GridSearchCV
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('svc', model)])

    # # used randomsearchcv to find the best params to use for SVC
    # # Best Hyperparameters: {'pca__n_components': 309, 'svc__C': 9.799098521619943, 'svc__gamma': 'auto', 'svc__kernel': 'rbf'}
    # # # Define hyperparameter distributions
    # parameters = {
    #     'pca__n_components': list(range(1, min(X_train.shape[0], X_train.shape[1]) + 1)),
    #     'svc__C': uniform(0.1, 10),        # Regularization parameter
    #     'svc__kernel': ['linear', 'rbf'],  # Kernel type
    #     'svc__gamma': ['scale', 'auto']    # Kernel coefficient for 'rbf'
    # }
    # # # Initialize RandomizedSearchCV
    # random_search = RandomizedSearchCV(pipe, parameters, n_iter=5, cv=5, scoring='accuracy', random_state=random_state, verbose=2)
    # # Print the best hyperparameters
    # print("Best Hyperparameters:", random_search.best_params_)

    # Train model
    pipe.fit(X_train, y_train)

    # Validate model
    predictions = pipe.predict(X_valid)

    # Print validation scores
    accuracy = accuracy_score(y_valid, predictions)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_valid, predictions))

    # Get model performance
    precision = precision_score(y_valid, predictions)
    recall = recall_score(y_valid, predictions)
    f1 = f1_score(y_valid, predictions)
    
    return pipe, accuracy, precision, recall, f1
