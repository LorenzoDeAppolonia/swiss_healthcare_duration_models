from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def scale_features(X_train, X_test, type='standard'):
    """
    Scale features using various scalers.

    Parameters:
    - X_train (pd.DataFrame): Features of the training set.
    - X_test (pd.DataFrame): Features of the test set.
    - type (str): Type of scaler ('standard', 'robust', 'minmax').

    Returns:
    - tuple: Scaled X_train and X_test.
    """
    if type == 'standard':
        sc_X = StandardScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
    elif type == 'robust':
        sc_X = RobustScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
    elif type == 'minmax':
        sc_X = MinMaxScaler()
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
    else:
        raise ValueError("Invalid scaling type. Use 'standard', 'robust', or 'minmax'.")

    return X_train_scaled, X_test_scaled




from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

plt.figure(figsize=(10, 10))

def plot_learning_curves(model, X_train, y_train, X_test, y_test, model_type, c):

  train_errors, test_errors = [], []

  for m in range(1, len(X_train)):

    model.fit(X_train[:m], y_train[:m])
    y_train_predict = model.predict(X_train[:m])
    y_test_predict = model.predict(X_test)

    train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
    test_errors.append(mean_squared_error(y_test, y_test_predict))

  plt.plot(np.sqrt(train_errors), 'b', linewidth=2, label="train_"+model_type)
  plt.plot(np.sqrt(test_errors), 'g' , linewidth=3, label="test_"+model_type)



# TODO: LOGISTIC REGRESSION

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap





def plot_decision_boundary(X_set, y_set, classifier, title):
    """
    Plot the decision boundary for the logistic regression model.

    Parameters:
    - X_set (pd.DataFrame): Features of the dataset.
    - y_set (pd.Series): Target variable of the dataset.
    - classifier (LogisticRegression): Fitted Logistic Regression model.
    - title (str): Plot title.
    """
    feature_count = X_set.shape[1]
    if feature_count == 2:
        X1, X2 = np.meshgrid(
            np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
            np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
        )
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X2.max())
        plt.ylim(X2.min(), X1.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    else:
        print("Plotting is supported for 2 features only.")

def visualize_results(X_train, y_train, X_test, y_test, classifier):
    """
    Visualize the training and test set results.

    Parameters:
    - X_train (pd.DataFrame): Features of the training set.
    - y_train (pd.Series): Target variable of the training set.
    - X_test (pd.DataFrame): Features of the test set.
    - y_test (pd.Series): Target variable of the test set.
    - classifier (LogisticRegression): Fitted Logistic Regression model.
    """
    # Visualizing the training set results
    plot_decision_boundary(X_train, y_train, classifier, title='Logistic Regression (Training Set)')

    # Visualizing the test set results
    plot_decision_boundary(X_test, y_test, classifier, title='Logistic Regression (Test Set)')

def histplot(df, title=None):
    fig, axes = plt.subplots(7, 3, figsize=(30, 24))
    fig.suptitle(title, y=1.02)
    sns.histplot(df["case_id"], ax=axes[0,0])
    sns.histplot(df["Spitex_Institute_code"], ax=axes[1,0])
    sns.histplot(df["Spitex_type_code"], ax=axes[2,0])
    sns.histplot(df["District_code"], ax=axes[3,0])
    sns.histplot(df["City_code"], ax=axes[0,1])
    sns.histplot(df["Available_nurse"], ax=axes[0,2])
    sns.histplot(df["Treatment"], ax=axes[1,1])
    sns.histplot(df["Team_code"], ax=axes[1,2])
    sns.histplot(df["Team_location_code"], ax=axes[2,1])
    sns.histplot(df["Team_members"], ax=axes[2,2])
    sns.histplot(df["patient_id"], ax=axes[3,1])
    sns.histplot(df["City_code_patient"], ax=axes[4,0])
    sns.histplot(df["Type_admission"], ax=axes[4,1])
    sns.histplot(df["Severity"], ax=axes[4,2])
    sns.histplot(df["Patient_network"], ax=axes[5,0])
    sns.histplot(df["Age"], ax=axes[5,1])
    sns.histplot(df["Deposit"], ax=axes[5,2])
    sns.histplot(df["Duration"], ax=axes[6,0])

    plt.show()

def boxplot(df, title=None):
    fig, axes = plt.subplots(7, 3, figsize=(30, 24))
    fig.suptitle(title, y=1.02)
    sns.boxplot(df["Spitex_Institute_code"], ax=axes[1,0])
    sns.boxplot(df["Spitex_type_code"], ax=axes[2,0])
    sns.boxplot(df["District_code"], ax=axes[3,0])
    sns.boxplot(df["City_code"], ax=axes[0,1])
    sns.boxplot(df["Available_nurse"], ax=axes[0,2])
    sns.boxplot(df["Treatment"], ax=axes[1,1])
    sns.boxplot(df["Team_code"], ax=axes[1,2])
    sns.boxplot(df["Team_location_code"], ax=axes[2,1])
    sns.boxplot(df["Team_members"], ax=axes[2,2])
    sns.boxplot(df["City_code_patient"], ax=axes[4,0])
    sns.boxplot(df["Type_admission"], ax=axes[4,1])
    sns.boxplot(df["Severity"], ax=axes[4,2])
    sns.boxplot(df["Patient_network"], ax=axes[5,0])
    sns.boxplot(df["Age"], ax=axes[5,1])
    sns.boxplot(df["Deposit"], ax=axes[5,2])
    sns.boxplot(df["Duration"], ax=axes[6,0])

    plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def optimize_hyperparameter_C(X_train, y_train, C_values, cv_folds):
    """
    Optimize the hyperparameter C for LogisticRegression with L2 regularization.

    Parameters:
    X_train: features of the training data
    y_train: target variable of the training data
    C_values: list of values for C to try
    cv_folds: number of cross-validation folds

    Returns:
    A fitted GridSearchCV object.
    """
    
    # Initialize the Logistic Regression estimator
    # Note that by default LogisticRegression uses L2 regularization
    log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    
    # Create the parameter grid
    param_grid = {'C': C_values}
    
    # Initialize GridSearchCV with the Logistic Regression estimator
    grid_search = GridSearchCV(log_reg, param_grid, cv=cv_folds, scoring='accuracy', verbose=1)
    
    # Fit the GridSearchCV to find the best C value
    grid_search.fit(X_train, y_train)
    
    return grid_search


def plot_accuracy_from_grid_search(grid_search):
    """
    Plots the mean cross-validated accuracy for each value of C from a fitted GridSearchCV object.

    Parameters:
    grid_search: The fitted GridSearchCV object from which to plot accuracies.
    """
    # Extract the mean test scores and the parameter settings
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    C_values = grid_search.cv_results_['param_C']

    # Plot the mean test scores
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, mean_test_scores, marker='o')
    
    # Log scale for C values since they can range over several orders of magnitude
    plt.xscale('log')
    
    plt.xlabel('Value of C')
    plt.ylabel('Cross-Validated Accuracy')
    plt.title('Accuracy for different values of C')
    plt.grid(True)
    plt.show()

    

def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fit the model with the training data --> potentially no need to fit
    model.fit(X_train, y_train)

    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Accuracy Evaluation
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print(f"Train Accuracy: {train_accuracy:.2f}")

    # wighted f-1 score evaluation
    weighted_f1_test = f1_score(y_test, y_pred_test, average='weighted')
    print(f"Weighted F1 Score (Test): {weighted_f1_test:.2f}")
    weighted_f1_train = f1_score(y_train, y_pred_train, average='weighted')
    print(f"Weighted F1 Score (Train): {weighted_f1_train:.2f}")


    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Reports
    class_report_train = classification_report(y_train, y_pred_train)
    print("Classification Report for Training Set:\n", class_report_train)
    class_report_test = classification_report(y_test, y_pred_test)
    print("Classification Report for Test Set:\n", class_report_test)


import pickle

def save_or_load_model(action, model=None, file_path=None):
    """
    Save or load a model using pickle.

    :param action: 'save' to save a model, 'load' to load a model
    :param model: the model to save (required if action is 'save')
    :param file_path: the file path to save to or load from
    :return: the loaded model if action is 'load'
    """
    if action == 'save' and model is not None and file_path is not None:
        # Save the model to the specified file path
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {file_path}")
    elif action == 'load' and file_path is not None:
        # Load the model from the specified file path
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return loaded_model
    else:
        raise ValueError("Invalid usage. Ensure correct action and parameters are provided.")

# Example usage:
# To save a model:
# save_or_load_model('save', model=my_model, file_path='my_model.pkl')

# To load a model:
# loaded_model = save_or_load_model('load', file_path='my_model.pkl')
import pickle

def save_or_load_model(action, model=None, file_path=None):
    """
    Save or load a model using pickle.

    :param action: 'save' to save a model, 'load' to load a model
    :param model: the model to save (required if action is 'save')
    :param file_path: the file path to save to or load from
    :return: the loaded model if action is 'load'
    """
    if action == 'save' and model is not None and file_path is not None:
        # Save the model to the specified file path
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {file_path}")
    elif action == 'load' and file_path is not None:
        # Load the model from the specified file path
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)
        print(f"Model loaded from {file_path}")
        return loaded_model
    else:
        raise ValueError("Invalid usage. Ensure correct action and parameters are provided.")

# Example usage:
# To save a model:
# save_or_load_model('save', model=my_model, file_path='my_model.pkl')

# To load a model:
# loaded_model = save_or_load_model('load', file_path='my_model.pkl')
