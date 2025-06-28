"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import load_and_plot_training_digits, train_classifier, predict_digits, visualize_predictions, generate_reports, rebuild_report_from_confusion_matrix

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits, data, X_train, X_test, y_train, y_test = load_and_plot_training_digits()



###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images


# Create a classifier: a support vector classifier
clf = train_classifier(X_train, y_train)




# Predict the value of the digit on the test subset
predicted = predict_digits(clf, X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

visualize_predictions(X_test, predicted)

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

generate_reports(clf, y_test, predicted)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

visualize_confusion_matrix(y_test, predicted)

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:

# Generate classification report and confusion matrix
conf_matrix = generate_reports(clf, y_test, predicted)

# The ground truth and predicted lists
y_true, y_pred, cm = rebuild_report_from_confusion_matrix(conf_matrix)

rebuild_report_from_confusion_matrix(y_true, y_pred)
