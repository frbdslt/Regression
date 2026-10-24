import numpy as np

def PrepareData(df):
    FeatureCols = [
    'Battery Voltage [V]',
    'Battery Current [A]',
    'Ambient Temperature [°C]',
    'Time [s]'
]
    TargetCol = 'Battery Temperature [°C]'

    X = df[FeatureCols].to_numpy(dtype=np.float64)
    y = df[TargetCol].to_numpy(dtype=np.float64).reshape(-1,1)

    return X,y

def StandardizeFeatures(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)

    std[std==0] = 1.0

    XNorm = (X-mean) / std

    return XNorm,mean,std

def TrainTestSplit(X,y,test_ratio=2.0,seed=42):
    np.random.seed(seed)
    total_samples = X.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    test_size = int(total_samples * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx],X[test_idx],y[train_idx],y[test_idx]
