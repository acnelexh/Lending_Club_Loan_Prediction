from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Hyperparameters for SVM and Logistic Regression
svm_param_distributions = {
    'C': np.logspace(-3, 3, 5)
}

logistic_param_distributions = {
    'C': np.logspace(-3, 3, 5),
    'penalty': ['l1', 'l2']
}

def preprocess_features(numerical, categorical, categorical_ordinal, labels):
    """
    Prepares and combines numerical, categorical, and ordinal feature arrays for use in 
    machine learning models. Also formats the labels for the model.

    Parameters:
    numerical (array-like): Numerical features.
    categorical (sparse matrix or array-like): Pre-processed categorical features.
    categorical_ordinal (array-like): Ordinal features with a meaningful order.
    labels (array-like): Target labels for the model.

    Returns:
    X (numpy.ndarray): Combined feature array.
    y (numpy.ndarray): Labels array, converted to float32.
    """
    numerical_feature = np.array(numerical)
    if sparse.issparse(categorical):
        categorical_feature = categorical.toarray()
    else:
        categorical_feature = np.array(categorical)
    categorical_ordinal_feature = np.array(categorical_ordinal)
    X = np.concatenate([numerical_feature, categorical_feature, categorical_ordinal_feature], axis=1)
    y = np.array(labels).astype(np.float32)
    return X, y

def reduce_dimensionality(X, n_components=50):
    """
    Reduces the dimensionality of the feature set using Truncated SVD and plots the 
    explained variance to help identify how many components are necessary to describe the data.

    Parameters:
    X (numpy.ndarray): The feature set to reduce.
    n_components (int): The desired number of dimensions.

    Returns:
    X_reduced (numpy.ndarray): The dimensionally reduced feature set.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    
    return X_reduced


def perform_random_search_svm(X_train, y_train, parameter_distributions, n_iter=5, cv=3):
    """
    Conducts a random search to find the best hyperparameters for an SVM model based 
    on precision score, using cross-validation.

    Parameters:
    X_train (numpy.ndarray): Training feature set.
    y_train (numpy.ndarray): Training labels.
    parameter_distributions (dict): Dictionary with parameters names (str) as keys 
                                    and distributions or lists of parameters to try.
    n_iter (int): Number of parameter settings that are sampled.
    cv (int): Number of folds in cross-validation.

    Returns:
    best_params_ (dict): Parameter setting that gave the best results on the hold out data.
    best_estimator_ (estimator): Estimator that was chosen by the search.
    """
    svm = LinearSVC(random_state=42, dual=False, max_iter=5000, tol=1e-2)
    random_search_svm = RandomizedSearchCV(
        svm, parameter_distributions, 
        n_iter=n_iter, 
        cv=cv, 
        n_jobs=-1, 
        scoring='precision', 
        random_state=42, 
        verbose=2
    )
    random_search_svm.fit(X_train, y_train)
    return random_search_svm.best_params_, random_search_svm.best_estimator_

def perform_random_search_logistic(X_train, y_train, parameter_distributions, n_iter=3, cv=2):
    """
    Performs a random search to optimize hyperparameters for a logistic regression model.

    Parameters:
    X_train (numpy.ndarray): Training data features.
    y_train (numpy.ndarray): Training data labels.
    parameter_distributions (dict): Range of parameters to sample during search.
    n_iter (int): Number of parameter settings to sample.
    cv (int): Number of folds for cross-validation.

    Returns:
    best_params_ (dict): Best hyperparameters found during search.
    best_estimator_ (LogisticRegression): Trained LogisticRegression model with best parameters.
    """
    logistic = LogisticRegression(random_state=42, max_iter=5000, solver='saga')
    random_search_logistic = RandomizedSearchCV(
        logistic, 
        parameter_distributions, 
        n_iter=n_iter,  
        cv=cv,  
        n_jobs=-1,  
        scoring='precision', 
        random_state=42,
        verbose=2  
    )
    random_search_logistic.fit(X_train, y_train)
    return random_search_logistic.best_params_, random_search_logistic.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of a model on a test set and calculate various metrics.

    Args:
        model: object
            The trained machine learning model to be evaluated.
        X_test: numpy array or pandas DataFrame
            The test data with features.
        y_test: numpy array or pandas Series
            The true labels for the test data.

    Returns:
        metrics: dict
            A dictionary containing various evaluation metrics, including accuracy, precision, recall, F1 score, ROC AUC, average precision, and confusion matrix.
        y_pred: numpy array
            The predicted labels on the test data.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'ap': average_precision_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    for metric, value in metrics.items():
        if isinstance(value, np.ndarray):
            value = np.mean(value)
        print(f'{metric.capitalize()}: {float(value):.3f}')
    
    return metrics, y_pred

def visualize_feature_importances_svm(model, feature_count):
    """
    Displays a bar chart of the feature importances for an SVM model.

    Parameters:
    model (LinearSVC): The trained SVM model.
    feature_count (int): The number of features in the model.
    """
    importances = abs(model.coef_[0])
    plt.subplots(figsize=(8, 5))
    plt.bar(range(feature_count), importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Weights of the Best SVM Model")
    plt.show()

def visualize_coefficients_logistic(model, feature_count):
    """
    Plots the coefficients of a logistic regression model to show their relative importance.

    Parameters:
    model (LogisticRegression): A trained logistic regression model.
    feature_count (int): The number of features used in the model.

    The function plots a bar graph with the feature indices on the x-axis and the 
    corresponding coefficients on the y-axis, indicating the influence of each feature.
    """
    coefficients = model.coef_[0]
    plt.subplots(figsize=(8, 5))
    plt.bar(range(feature_count), coefficients)
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.title("Coefficients of the Logistic Regression Model")
    plt.show()
