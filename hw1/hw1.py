from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import os

# ======================== PREPROCESSING ========================
def get_idx2name(label_info):
    """ Returns a dictionary with the key as encoded index and value as name """
    idx2name = {}
    with open(label_info) as f:
        f.readline() # header will not be read in
        for line in f:
            idx, name = line.rstrip().split()
            idx2name[idx] = name
    
    return idx2name

def prepare_data(gtlabels, label_info, subset_names):
    """ Returns X, y given by subset, where X contains image features and y is the new index """

    idx2name = get_idx2name(label_info)

    X = []
    y = []
    with open(gtlabels) as f:
        for line in f:
            fname, encoding = line.split(' ', 1)
            Xi = np.load(os.path.join('imagefeatures', fname + '_ft.npy'))
            yi = np.array([int(i) for i in encoding.split(' ')])

            for idx, boolean in enumerate(yi):
                if idx2name[str(idx)] in subset_names and boolean == 1:
                    X.append(Xi)
                    y.append(idx)

    # Reset the index based on subset
    labels = sorted(set(y))
    y = [labels.index(yi) for yi in y]

    return np.array(X), np.array(y)

def split_data(X, y, splits, verbose=True):
    """ Splits the data into train, val and test in a stratified manner """

    # normalize the ratios of the splits
    train_ratio, val_ratio, test_ratio = np.asarray(splits) / np.sum(splits)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio/(train_ratio + val_ratio), stratify=y_trainval, random_state=42)
    
    if verbose:
        # print the data split shapes
        print("X_train shape: {}".format(X_train.shape))
        print("y_train shape: {}".format(y_train.shape))
        print("X_val shape: {}".format(X_val.shape))
        print("y_val shape: {}".format(y_val.shape))
        print("X_test shape: {}".format(X_test.shape))
        print("y_test shape: {}".format(y_test.shape))

        # print the number of classes for each split 
        print("Number of classes in y_train...")
        print_class_counts(y_train)
        print("Number of classes in y_val...")
        print_class_counts(y_val)
        print("Number of classes in y_test...")
        print_class_counts(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def save_npy(array, fnames):
    """ Save numpy arrays """
    for arr, fname in zip(array, fnames):
        np.save(fname + '.npy', arr)

def load_npy(fnames):
    """ Load numpy arrays """
    return [np.load(fname + '.npy') for fname in fnames]

# ============================ MODEL ============================
class OVRSVC:
    def __init__(self, c, kernel):
        self.c = c
        self.kernel = kernel
        self.models = None

    def fit(self, X, y):
        """ Fits X and y to each SVM model of the OVRSVC classifier """
        self.n_classes = len(np.unique(y))
        self.models = [SVC(kernel=self.kernel, C=self.c, probability=True) for i in range(self.n_classes)]
        
        y_onehot = pd.get_dummies(y).to_numpy()
        for i in range(self.n_classes):
            # col 0: spring, col 1: summer, col 2: autumn
            self.models[i].fit(X, y_onehot[:,i])
    
    def predict(self, X):
        """ Predicts the class-label as the index of the SVM with the highest prediction score """
        y_pred = []
        scores = [model.predict_proba(X) for model in self.models]

        for i in range(len(X)):
            svm_idx = np.argmax([scores[j][i][1] for j in range(len(scores))])
            y_pred.append(svm_idx)

        return np.array(y_pred)

def vanilla_acc(y_pred, y_true):
    """ Computes the ratio of predicted labels in y_pred that match exactly the corresponding labels in y_true """
    assert(len(y_pred) == len(y_true))
    return np.sum(y_pred==y_true) / len(y_true)

def class_avg_acc(y_pred, y_true):
    """ Computes class-wise average accuracy """
    assert(len(y_pred) == len(y_true))
    n_classes = len(np.unique(y_true))

    scores = []
    for i in range(n_classes):
        true_positives = 0
        total_positives = 0
        for yi_pred, yi_true in zip(y_pred, y_true):
            if yi_true == i:
                total_positives += 1
                if yi_pred == i:
                    true_positives += 1
        
        score = 0 if true_positives == 0 else true_positives / total_positives
        scores.append(score)

    return sum(scores) / n_classes

def reg_tuning(X_train, y_train, X_val, y_val, reg_constants, kernel):
    """ Returns the best value for C based on the performance on the validation set """
    val_scores = []
    for c in reg_constants:
        # Train the model
        model = OVRSVC(c, kernel)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)
        val_score = class_avg_acc(y_pred, y_val)
        val_scores.append(val_score)

        print("c: {}. val_score: {}".format(c, val_score))

    return reg_constants[np.argmax(val_scores)]

# ======================= UTLITY FUNCTIONS ======================= 

def print_report(y_pred, y_true):
    print("Vanilla accuracy: {}".format(vanilla_acc(y_pred, y_true)))
    print("Class-wise average accuracy: {}\n".format(class_avg_acc(y_pred, y_true)))

def print_class_counts(arr):
    unique_elements, counts_elements = np.unique(arr, return_counts=True)
    for element, count in zip(unique_elements, counts_elements):
        print("{}: {}".format(element, count))

# ======================== MAIN FUNCTION ========================
if __name__ == '__main__':
    # User-defined variables
    seasons = ['Spring', 'Summer', 'Autumn']
    gtlabels_fpath = 'gtlabels.txt'
    label_info_fpath = 'label_info.txt'
    kernels = ['linear', 'poly', 'rbf']
    reg_constants = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10]

    # Prepare X, y
    X, y = prepare_data(gtlabels_fpath, label_info_fpath, seasons)
    assert len(X) == len(y) == 1145 # specified in handout

    # Split the data and save the numpy files
    fnames = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
    save_npy(split_data(X, y, splits=[0.65, 0.15, 0.20]), fnames)

    # Load the numpy files
    X_train, y_train, X_val, y_val, X_test, y_test = load_npy(fnames)

    # Loop through each kernel
    for kernel in kernels:
        print("======= SVC with {} kernel =======".format(kernel))
        # Hyperparameter tuning of the reg constant
        best_reg = reg_tuning(X_train, y_train, X_val, y_val, reg_constants, kernel)

        # Train on train data only, and report performance
        clf = OVRSVC(best_reg, kernel)
        clf.fit(X_train, y_train)

        print("Model fitted on train data only...")
        print("[Validation Results]")
        y_val_pred = clf.predict(X_val)
        print_report(y_val_pred, y_val)

        print("[Test Results]")
        y_test_pred = clf.predict(X_test)
        print_report(y_test_pred, y_test)

        # Train on train + val data, and report performance
        X_trainval = np.concatenate((X_train, X_val))
        y_trainval = np.concatenate((y_train, y_val))
        
        clf = OVRSVC(best_reg, kernel)
        clf.fit(X_trainval, y_trainval)

        print("Model fitted on train + val data...")
        print("[Validation Results]")
        y_val_pred = clf.predict(X_val)
        print_report(y_val_pred, y_val)

        print("[Test Results]")
        y_test_pred = clf.predict(X_test)
        print_report(y_test_pred, y_test)