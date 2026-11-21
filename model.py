"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    X = np.array(X, dtype=float, copy=True)
    for i in range(X.shape[1]):
        x_sum = 0.0
        x_num = 0

        for j in range(X.shape[0]):
            if not np.isnan(X[j][i]):
                x_sum = x_sum + X[j][i]
                x_num = x_num + 1
        if x_num == 0:
            mean = 0.0
        else:
            mean = x_sum/x_num
            
        for j in range(X.shape[0]):
            if np.isnan(X[j][i]):
                X[j][i]=mean
    return X

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    X = np.asarray(X)
    lowers = np.zeros(X.shape[1])
    uppers = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        q1 = np.percentile(X[:,i], 25, axis=0)
        q3 = np.percentile(X[:,i], 75, axis=0)
        iqr=q3-q1
        lower = q1-k*iqr
        upper = q3+k*iqr
        lowers[i]=lower
        uppers[i]=upper
    return (lowers,uppers)

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X,lower,upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    denominator = denominator+eps
    return numerator/denominator

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    
    col = np.asarray(col).reshape(-1,1)
    return np.concatenate([X,col],axis=1)
    ## OR use column stack np.column_stack([X,col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    n = labels.shape[0]
    unique_labels = np.unique(labels)
    m = unique_labels.shape[0]
    one_hot_encodes = np.zeros((n,m))
    mask = labels[:,None]==unique_labels[None,:]
    unique_labels=np.where(mask,1,0)
    return unique_labels.astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    X = np.asarray(X, dtype=float)
    means = np.mean(X,axis=0)
    stds = np.std(X, axis=0)
    mask = stds==0.0
    stds = np.where(mask,1.0,stds)
    return means,stds

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    x = np.asarray(X,dtype=float).copy()
    return (x-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    X = np.asarray(X, dtype=float)
    tmp = np.ones((X.shape[0],1)) # both tmp and X should have ndim=2
    return np.hstack([tmp,X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    tmp = np.arange(n_samples)
    selected = np.random.RandomState(seed).permutation(n_samples)
    return selected

# Step 11 - partition_indices
import math
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    train_indices = indices[:math.floor(n*train_ratio)]
    validation_indices = indices[math.floor(n*train_ratio):math.floor(n*(train_ratio+val_ratio))]
    test_indices = indices[math.floor(n*(train_ratio+val_ratio)):]
    return train_indices,validation_indices,test_indices

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices],y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # Taking the gradient of MSE and setting it to zero gives (X.T@X)@W=X.T@Y
    X = np.asarray(X,dtype=float)
    y = np.asarray(y, dtype=float)
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    
    ## This only works for invertible matrix
    # left = X.T@X
    # right = X.T@y
    # theta = np.linalg.solve(left,right)
    return theta

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    X = np.asarray(X,dtype=float)
    theta = np.asarray(theta,dtype=float)
    prediction = X@theta.T
    return prediction

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_true-y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(np.mean((y_true-y_pred)**2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true,dtype=float)

    SS_res = np.sum((y_pred-y_true)**2)
    y_true_mean = np.mean(y_true)
    SS_tot = np.sum((y_true-y_true_mean)**2)
    if SS_tot==0:
        return 0.0
    else: 
        return 1-(SS_res/SS_tot)

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    y_pred = np.asarray(y_pred,dtype=float)
    y_true = np.asarray(y_true,dtype=float)
    r = y_true - y_pred
    mean_r = np.mean(r)
    std_r = np.std(r)
    median_abs_r = np.median(np.abs(r))
    return {'mean':mean_r,'std':std_r,'median_abs':median_abs_r}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X = np.asarray(X, dtype=float)
    clean_x = impute_nan_with_mean(X)
    lowers,uppers = compute_iqr_bounds(clean_x, iqr_k)
    clean_x = clip_columns(clean_x,lowers,uppers)
    return clean_x

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    numerators = X_num[:,ratio_num_idx]
    denominators = X_num[:,ratio_den_idx]
    ratio = make_ratio_feature(numerators,denominators)
    appended = append_column(X_num,ratio)
    if cat_labels is None:
        return appended
    else:
        encoded = one_hot_encode(cat_labels)
        return np.hstack([appended,encoded])

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n_samples = X.shape[0]
    indices = make_shuffled_indices(n_samples,seed)
    train_indices, validation_indices, test_indices = partition_indices(indices, train_ratio, val_ratio)
    X_train, y_train = subset_xy(X,y,train_indices)
    X_val, y_val = subset_xy(X,y,validation_indices)
    X_test, y_test = subset_xy(X,y,test_indices)
    return {'X_train':X_train,'y_train':y_train,'X_val':X_val,'y_val':y_val,'X_test':X_test,'y_test':y_test}

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    train_mean, train_std = fit_standardizer(splits['X_train'])
    X_train_standardized = apply_standardizer(splits['X_train'],train_mean,train_std)
    X_val_standardized = apply_standardizer(splits['X_val'],train_mean,train_std)
    X_test_standardized = apply_standardizer(splits['X_test'],train_mean,train_std)
    X_train_final = add_bias_column(X_train_standardized)
    X_val_final = add_bias_column(X_val_standardized)
    X_test_final = add_bias_column(X_test_standardized)
    std_splits = {'X_train':X_train_final,'y_train':splits['y_train'],'X_val':X_val_final,'y_val':splits['y_val'],'X_test':X_test_final,'y_test':splits['y_test']}
    return std_splits, train_mean, train_std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    mae = mean_absolute_error(y_true,y_pred)
    rmse = root_mean_squared_error(y_true,y_pred)
    r2 = r_squared(y_true,y_pred)
    residual_summary_ = residual_summary(y_true,y_pred)
    return {'mae':mae,'rmse':rmse,'r2':r2,'residual_summary':residual_summary_}

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X = prepare_cleaned_features(X,iqr_k)
    X = assemble_feature_matrix(X,ratio_num_idx,ratio_den_idx,cat_labels)
    splits = make_train_val_test(X,y,train_ratio,val_ratio,seed)
    splits, mean, std = standardize_and_add_bias(splits)
    left = splits['X_train'].T@splits['X_train']
    right = splits['X_train'].T@splits['y_train']
    theta = ols_fit(splits['X_train'],splits['y_train'])
    predictions_test = ols_predict(splits['X_test'],theta)
    predictions_val = ols_predict(splits['X_val'],theta)
    val_metrics = evaluate_predictions(splits['y_val'],predictions_val)
    test_metrics = evaluate_predictions(splits['y_test'],predictions_test)
    return {'theta':theta,'y_test':splits['y_test'],'y_test_pred':predictions_test,'test_metrics':test_metrics,'val_metrics':val_metrics}

