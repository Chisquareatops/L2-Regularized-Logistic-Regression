# IMPORTS
# these packages must be installed for this code to run
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scipy.linalg
import matplotlib.pyplot as plt

# Load the spam data set
# rows are 4601 emails with 57 attributes, plus 1 class label designating the email spam or not spam
# 1813 (39.4%) are spam
spam = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', header=0, delim_whitespace=True) 
spam = spam.dropna()

# Transform the data set into a matrix of features (X) and the corresponding label (Y)
X = spam.drop(spam.columns[-1], axis=1)
X = pd.get_dummies(X, drop_first=True) 
Y = spam.iloc[:,-1]

# Split the data into training and test sets
# 25% of the data will go into the test set and 75% into the training set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1987)

# Standardize and scale the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def computegrad(beta, lamb, X, Y):
    """
    Computes the gradient of the l2-regularized logistic regression objective function
    when passed the following parameters:
    :param beta: array containing beta values
    :param lamb: constant value for regularization
    :param X: data frame containing features
    :param Y: data frame containing labels
    """
    n = X.shape[0]
    X_beta = X.dot(beta)
    frac_term = 1 / (1 + np.exp(Y * X_beta))
    frac = -1 / n * (X.T.dot(Y * frac_term))
    gradient = (frac + (lamb * beta * 2))
    return (gradient)

def objective(beta, lamb, X, Y):
    """
    A helper function to calculate the objective value
    when passed the following parameters:
    :param beta: array containing beta values
    :param lamb: constant value for regularization
    :param X: data frame containing features
    :param Y: data frame containing labels
    """
    n = X.shape[0]
    one_n = (1 / n)
    inner_term_1 = X.dot(beta)
    inner_term_2 = (-Y).dot(inner_term_1)
    exp_term = np.exp(inner_term_2)
    log_term = (np.log(exp_term + 1))
    sum_term = np.sum(log_term)
    term_1 = (one_n * sum_term)
    
    exp_term_2 = (np.linalg.norm(beta)**2)
    term_2 = (lamb * exp_term_2)
    
    objective = (term_1 + term_2)
    return (objective)

def backtrack(beta, lamb, eta, max_iter, X, Y):
    """
    Implements the backtracking rule to find the optimal step size
    when passed the following parameters:
    :param beta: array containing beta values
    :param lamb: constant value for regularization
    :param eta: constant, the step size to be adjusted via backtracking
    :param max_iter: constant, max number of iterations for the backtracking algorithm
    :param X: data frame containing features
    :param y: data frame containing labels    
    """
    alpha = 0.5 # also tried 0.5
    beta_2 = 0.8
    grad_beta = computegrad(beta, lamb, X, Y)
    norm_beta = np.linalg.norm(grad_beta)
    eta_new = 0
    t = 0
    objective_1 = objective(beta-eta*grad_beta, lamb, X, Y)
    objective_2 = objective(beta, lamb, X, Y)
    extra_term = (alpha * eta* norm_beta**2)
    while (eta_new == 0 and t < max_iter):
        if (objective_1 < (objective_2 - extra_term)):
            eta_new = 1
        else:
            eta = (eta * beta_2)
            t += 1
    return eta

def fastgradalgo(beta, eta_init, lamb, max_iter, X, Y):
    """
    Uses fast gradient descent to minimize the objective function
    when passed the following parameters:
    :param beta: array containing beta values
    :param eta_init: constant, the step size to be adjusted via backtracking
    :param lamb: constant value for regularization
    :param max_iter: constant, max number of iterations for the backtracking algorithm
    :param X: data frame containing features
    :param y: data frame containing labels 
    """
    beta_init = beta
    theta = beta
    grad_theta = computegrad(theta, lamb, X, Y)
    beta_vals = [beta]
    theta_vals = [beta]
    eta = eta_init
    t = 0
    while (t < max_iter):
        eta = backtrack(theta, lamb, eta, 1000, X, Y)
        beta_temp = computegrad(theta_vals[t], lamb, X, Y)
        beta_new = (theta_vals[t] - (eta * beta_temp))
        beta_vals.append(np.array(beta_new))
        
        t_term = (t / (t+3))
        theta_new = (beta_vals[t+1] + (t_term * (beta_vals[t+1] - beta_vals[t])))
        theta_vals.append(np.array(theta_new))
        
        beta = beta_new
        t += 1
    return (beta_vals)

# Set constants for convenience
lamb = 1
n = X_train.shape[0]
d = X_train.shape[1]
beta_init = np.zeros(d)

# Find an eta to start with
X_term = X_train.T.dot(X_train)
full_term = (1 / len(Y_train) * X_term)
eig = scipy.linalg.eigh(full_term, eigvals = (d-1, d-1), eigvals_only=True)[0] + lamb
eta_init = (1 / eig)

# Run the fast gradient descent algorithm using the eta found above and max iterations of 1000
betas = fastgradalgo(beta_init, eta_init, lamb, 1000, X_train, Y_train)
final_betas = betas[-1]

# Use the final betas to make the ultimate predictions
predictions = np.dot(X_test, final_betas)
predictions = preprocessing.binarize(predictions.reshape(-1,1), threshold=0)
predictions = preprocessing.label_binarize(np.ravel(predictions), [0,1], neg_label=0)
predictions = np.squeeze(predictions)

# Look at the predictions to confirm correct execution
print('predictions: ')
print(predictions)

# Use sklearn.metrics to find the accuracy
accuracy = accuracy_score(Y_test, predictions)
print('Accuracy: ')
print(accuracy)

# Use sklearn.metrics to create a confusion matrix. a confusion matrix is read:
# TRUE POSITIVES  |  FALSE POSITIVES
# FALSE NEGATIVES | TRUE NEGATIVES
conf_matrix = confusion_matrix(Y_test, predictions)
print('confusion matrix: ')
print(conf_matrix)

objecfast = []

for i in range(len(betas)):
    obj = objective(X=X_train, Y=Y_train, beta=betas[i], lamb=0.1)
    objecfast.append(obj)
    print(obj)
objecfast = pd.DataFrame(objecfast)
plt.plot(objecfast)
plt.title('Visualizing the Training Process')
plt.xlabel('Iteration')
plt.ylabel('Objective')