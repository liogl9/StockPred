import torch
import numpy as np

def create_dataset_one_output(dataset, lookback):
    """Transform a time series into a prediction dataset with the last day as target for the whole time samples
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(dataset.shape[0]-lookback):
        feature = dataset[i:i+lookback,:-1]
        target = dataset[i+lookback:i+lookback+1,-1:]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def create_dataset_whole_output(dataset, lookback):
    """Transform a time series into a prediction dataset with the next day as target for each time sample
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def create_single_sample(dataset, lookback):
    """Get the first sample of a datset with the next day as target for each time sample
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    if dataset.shape[1] == 1:
        feature = dataset[0:lookback]
        target = dataset[1:lookback]
        X.append(feature)
        y.append(target)
    else:
        feature = dataset[0:lookback,:-1]
        target = dataset[1:lookback,-1:]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def get_single_sample(dataset, lookback, pos):
    """Get the n sample of a dataset with the next day as target for each time sample
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    if dataset.shape[1] == 1:
        feature = dataset[pos:pos+lookback]
        target = dataset[pos+1:pos+lookback+1]
        X.append(feature)
        y.append(target)
    else:
        feature = dataset[pos:pos+lookback,:-1]
        target = dataset[pos+1:pos+lookback+1,-1:]
        X.append(feature)
        y.append(target)
    return torch.tensor(np.array(X)), torch.tensor(np.array(y))

def create_testset(dataset, lookback):
    X = []
    feature = dataset[0:lookback]
    X.append(feature)
    return torch.tensor(np.array(X))