
import numpy as np

def compute_TP(target, prediction, threshold):
    '''
    compute the  number of true positives
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    target = np.array([target>threshold])
    prediction = np.array([prediction>threshold])
    tp = np.logical_and(target, prediction)
    return tp.sum()

def compute_TN(target, prediction, threshold):
    '''
    compute the  number of true negatives
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    target = np.array([target<threshold])
    prediction = np.array([prediction<threshold])
    tn = np.logical_and(target, prediction)
    return tn.sum()

def compute_FP(target, prediction, threshold):
    '''
    compute the  number of  false positives
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    target = np.array([target<threshold])
    prediction = np.array([prediction>threshold])
    fp = np.logical_and(target, prediction)
    return fp.sum()

def compute_FN(target, prediction, threshold):
    '''
    compute the  number of false negatives
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    threshold: float
    '''

    target = np.array([target>threshold])
    prediction = np.array([prediction<threshold])
    fn = np.logical_and(target, prediction)
    return fn.sum()

