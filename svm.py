from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

param_distributions = {
    'C': np.logspace(-3, 3, 5), 
}

def preprocess_features(numerical, categorical, categorical_ordinal, labels):
    '''
    Convert to numpy array if necessary and concat
    args:
        numerical: List or numpy array
        categorical: scipy.sparse matrix or numpy array
        categorical_ordinal: List or numpy array
        labels: List or numpy array
    return:
        X: numpy array
        y: numpy array
    '''
        
    # Make sure numerical_feature is a numpy array
    numerical_feature = np.array(numerical)

    # If categorical_feature is a sparse matrix, convert it to a numpy array
    if sparse.issparse(categorical):
        categorical_feature = categorical.toarray()
    else:
        categorical_feature = np.array(categorical)

    # Make sure categorical_ordinal_feature is a numpy array
    categorical_ordinal_feature = np.array(categorical_ordinal)

    # Standardize the numerical features using StandardScaler
    scaler = StandardScaler()
    numerical_feature = scaler.fit_transform(numerical_feature)  # This line was missing

    # Now concatenate the arrays
    X = np.concatenate([numerical_feature, categorical_feature, categorical_ordinal_feature], axis=1)
    y = np.array(labels).astype(np.float32)

    return X, y

def reduce_dimensionality(X, n_components=100):
    '''
    Reduce the dimensionality of the data
    args:
        X: numpy array or pandas DataFrame
            The input data with features.
        n_components: int, optional (default=100)
            The number of components to reduce to.
    return:
        X_reduced: numpy array
            The dimensionality-reduced data.
    '''

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X)
    return X_reduced

def perform_random_search(X_train, y_train, parameter_distributions, n_iter=3, cv=2):
    '''
    Perform random search for hyperparameter tuning of a LinearSVC model.
    Args:
        X_train: numpy array or pandas DataFrame
            The training data with features.
        y_train: numpy array or pandas Series
            The training labels.
        parameter_distributions: dict
            A dictionary specifying the parameter distributions for RandomizedSearchCV.
        n_iter: int, optional (default=3)
            The number of parameter settings that are sampled.
        cv: int, optional (default=2)
            The number of cross-validation folds.
    Returns:
        best_params: dict
            The best hyperparameters found by random search.
        best_estimator: object
            The best estimator (LinearSVC model) obtained from random search.
    '''

    # Define the LinearSVC model with further simplified parameters
    svm = LinearSVC(random_state=42, dual=False, max_iter=5000, tol=1e-2)

    # Define RandomizedSearchCV for hyperparameter tuning with reduced iterations and folds
    random_search_svm = RandomizedSearchCV(
        svm, parameter_distributions, n_iter=n_iter, cv=cv, n_jobs=-1, scoring='precision', random_state=42
    )
    random_search_svm.fit(X_train, y_train)

    return random_search_svm.best_params_, random_search_svm.best_estimator_


def evaluate_model(model, X_test, y_test):
    '''
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
    '''
    # Evaluate performance on test set
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Metrics calculation
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
            # If the value is an array, calculate a summary statistic, e.g., mean
            value = np.mean(value)
        print(f'{metric.capitalize()}: {float(value):.3f}')
    
    return metrics, y_pred

def visualize_feature_importances(model, feature_count):
    '''
    Visualize the feature importances of a machine learning model.

    Args:
        model: object
            The trained machine learning model for which feature importances will be visualized.
        feature_count: int
            The number of features in the dataset.

    Returns:
        None
    '''

    importances = abs(model.coef_[0])
    plt.subplots(figsize=(8, 5))
    plt.bar(range(feature_count), importances)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("Feature Weights of the Best SVM Model")
    plt.show()