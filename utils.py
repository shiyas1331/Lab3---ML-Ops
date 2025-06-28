import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_and_plot_training_digits():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Plot first 4 training images
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Training: {label}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )
    return digits, data, X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train):
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf

def predict_digits(clf, X_test):
    return clf.predict(X_test)

def visualize_predictions(X_test, predicted):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

def generate_reports(clf, y_test, predicted):
    print(f"\nClassification report for classifier {clf}:\n")
    print(metrics.classification_report(y_test, predicted))

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()
    return disp.confusion_matrix

def rebuild_report_from_confusion_matrix(cm):
    y_true = []
    y_pred = []
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print("\nClassification report rebuilt from confusion matrix:\n")
    print(metrics.classification_report(y_true, y_pred))
