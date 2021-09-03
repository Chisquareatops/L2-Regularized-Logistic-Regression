This code is a slightly updated version of the fastgradalgo function from Homework 3.

It uses L2 regularized logistic regression and a backtracking function.

L2_Regularized_Spam.py uses the function to examine the Spam dataset from
https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data
The Spam dataset contains information about 4601 emails with 57 continuous attributes
plus 1 label designating the email 'spam' (1) or 'not spam' (0).
1813 (39.4%) of the emails are spam.

L2_Regularized_Synthetic.py uses the same functions to examine synthetic data
created within the file itself. The synthetic data is drawn randomly from a uniform 
distribution and contains two classes (50% class 1, 50% class 2).

L2_Regularized_Compare_to_SKL.py contains all the code from L2_Regularized_Spam.py
and uses built in functions from SciKit Learn to create a linear regression model
and fit it to the same data. This file gives the accuracy and a confusion matrix
for fastgradalgo and for the SKL model to compare them.

The following packages must be installed to execute the .py files in this folder:

numpy
pandas
sklearn
sklearn.model_selection
sklearn.metrics
scipy.linalg