# Logistic-Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable,
although many more complex extensions exist. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Logistic regression analysis is used to examine the association of (categorical or continuous) independent variable(s) with one dichotomous dependent variable.
This is in contrast to linear regression analysis in which the dependent variable
is a continuous variable.

**Confusion matrix** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the number of target classes. The matrix compares the actual 
target values with those predicted by the machine learning model.The rows represent the predicted values of the target variable.

Precision = TP/TP+FP   (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)

Recall = TP/TP+FN      (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

Precision and recall are two extremely important model evaluation metrics. While precision refers to the percentage of your results which are relevant, 
recall refers to the percentage of total relevant results correctly classified by your algorithm

Accuracy = Number of correct predictions / Total number of predictions.       (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

**Area under the curve(AUC):**

The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve. 
The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes.

https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py

**Libraries**

import numpy as np

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from sklearn.model_selection import train_test_split

