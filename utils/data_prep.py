
import numpy as np 
def calculate_scalar(x):

    if x.ndim <= 2:
        axis = 0

    elif x.ndim == 3:
        axis = (0, 1)

    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    maxi = np.max(x, axis=axis)

    return mean, std, maxi


def scale(x, mean, std):

    return (x - mean) / std


def inverse_scale(x, mean, std):

    return x * std + mean

def binarize(tensor, threshold=0.5):
    return ((tensor - threshold) > 0).astype('float')