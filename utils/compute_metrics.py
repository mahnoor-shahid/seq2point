
import numpy as np

def compute_TP(y_value, pred, threshold):
    '''
    compute the  number of true positives
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    y_value = np.array([y_value>threshold])
    pred = np.array([pred>threshold])
    tp = np.logical_and(y_value, pred)
    return tp.sum()

def compute_TN(y_value, pred, threshold):
    '''
    compute the  number of true negatives
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    y_value = np.array([y_value<threshold])
    pred = np.array([pred<threshold])
    tn = np.logical_and(y_value, pred)
    return tn.sum()

def compute_FP(y_value, pred, threshold):
    '''
    compute the  number of  false positives
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    y_value = np.array([y_value<threshold])
    pred = np.array([pred>threshold])
    fp = np.logical_and(y_value, pred)
    return fp.sum()

def compute_FN(y_value, pred, threshold):
    '''
    compute the  number of false negatives
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    y_value = np.array([y_value>threshold])
    pred = np.array([pred<threshold])
    fn = np.logical_and(y_value, pred)
    return fn.sum()

def compute_recall(y_value, pred, threshold):
    '''
    compute the recall rate
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    tp = compute_TP(y_value, pred, threshold)
    fn = compute_FN(y_value, pred, threshold)

    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def compute_precision(y_value, pred, threshold):
    '''
    compute the  precision rate
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    tp = compute_TP(y_value, pred, threshold)
    fp = compute_FP(y_value, pred, threshold)

    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def compute_f1(y_value, pred, threshold):
    '''
    compute the  F1 score
    Parameters:
    ----------------
    y_value: the groud truth , np.array
    pred: the pred, np.array
    threshold: float
    '''

    recall = compute_recall(y_value, pred, threshold)
    precision = compute_precision(y_value, pred, threshold)

    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1
