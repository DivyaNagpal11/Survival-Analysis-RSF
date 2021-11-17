import re
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from split._split import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import concordance_index_censored
from sklearn.feature_selection import VarianceThreshold

from sklearn.inspection import permutation_importance
from sklearn.utils import check_array, check_consistent_length


def pickl(file, thing=None):
    """Either pickle or unpickle a thing."""
    if thing is not None:
        pd.to_pickle(thing, file)
    else:
        return pd.read_pickle(file)
    

def clip_right_tail2(df, right_tail_end):
    """Returns the df- dataframe after removing all response time values greater than right_tail_end percentile value"""
    # Converting "response_time" column data into pandas "float" type.
    response_time_data = df["response_time"].astype("float")
    
    # Computing the right tail end value given right_tail_end percentile value.
    right_tail_end_value = response_time_data.quantile(right_tail_end)
    
    # Clipping the data above the "right_tail_end_value".
    df = df[df["response_time"]<=right_tail_end_value].reset_index(drop=True)
    return df


def cat_corr_filter(cat_df, binary_cat_stats):
    #We will use pearson correlation to filter cat binary columns:
    le = LabelEncoder()
    cat_binary = cat_df[binary_cat_stats.index].apply(lambda col: le.fit_transform(col.astype(str)), axis=0, result_type="expand")
    cat_binary = cat_binary.astype("object")
    ref_binary = cat_binary.describe().transpose()
    ref_binary = ref_binary[ref_binary["unique"]==2]
    cat_binary = cat_binary[ref_binary.index].astype('int')
    dropped, cat_binary= correlation(cat_binary, 0.80)
    #print("Binary Categorical features dropped due to correlation: {}'.format(dropped))
    return dropped


def filter_cat_cols(cat_df):
    """ Performs Feature Selection on Categorical Features based on Cardinality, Completeness etc.
    
    Args:
        cat_df: Claim filing indicator specific categorical columns dataframe.
    
    Returns:
        cat_stats: .describe output of the retained categorical columns that satisfies all the condition checks.
        filtered_cat_columns_tracker: A tracker dictionary that contains the condition name and a list of columns that are dropped due to that condition.
    
    """
    
    print("***Begin filter_cat_cols method***")
    
    # Dictionary to track filter conditions and the corresponding columns that are removed.
    filtered_cat_columns_tracker = {}
    
    # Converting all columns into pandas "object" type
    cat_df = cat_df.astype("object")
    
    # Saving pandas .describe() output in transposed form to cat_stats
    cat_stats = cat_df.describe().transpose()
    
    # Condition#1 - Remove a column if it has NaN values in all observations.
    filtered_cat_columns_tracker["columns_with_all_nan_values"] = list(cat_stats[cat_stats["count"]==0].index)
    cat_stats = cat_stats[cat_stats["count"]!=0]
    
    # Condition#2 - Remove a column if it has a single unique value in all observations
    mask = (cat_stats["count"]==cat_df.shape[0]) & (cat_stats["unique"]==1)
    filtered_cat_columns_tracker["columns_with_single_value"] = list(cat_stats[mask].index)
    cat_stats = cat_stats[~mask]
    
    # Condition#3 - Missingness: Remove a column if it has 90% or more missing values
    cat_stats["missingness"] = 100-(cat_stats["count"]/cat_df.shape[0]*100)
    mask = cat_stats["missingness"] >= 90
    filtered_cat_columns_tracker["missingness"] = list(cat_stats[mask].index)
    cat_stats = cat_stats[~mask]
    
    # Condition#4 - Remove a column if a single value under that column has 95% or more frequency
    cat_stats["top_freq"] = cat_stats["freq"]/cat_df.shape[0]*100
    mask = cat_stats["top_freq"] >= 95
    filtered_cat_columns_tracker["top_freq"] = list(cat_stats[mask].index)
    cat_stats = cat_stats[~mask]
    
    # Condition#5 - Remove a column if it has a cardinality of greater than 100
    mask = cat_stats["unique"] > 100
    filtered_cat_columns_tracker["cardinality"] = list(cat_stats[mask].index)
    cat_stats = cat_stats[~mask]

    # Sorting the cat_stats based on missingness and unique values
    cat_stats = cat_stats.sort_values(["missingness","unique"], ascending=False)
    
    # Condition#6 - Check and remove any correlated binary columns (columns with count==2)
    binary_cat_stats = cat_stats[cat_stats["unique"]==2]
    
    dropped_cols = cat_corr_filter(cat_df, binary_cat_stats)
    filtered_cat_columns_tracker["correlated"] = dropped_cols
    
    #Creating final df
    cat_df = cat_df[np.setdiff1d(cat_stats.index, list(dropped_cols))].astype("object")
    
    cat_stats = cat_df.describe().transpose()
    print("***End filter_cat_cols method***")
    print("***########################################***\n\n")
    return cat_stats, filtered_cat_columns_tracker


def variance_threshold_selector(num_df, threshold=0.1):
    """ This method identifies and drops the list of continuous features with a variance value less than that of the threshold value.
    
    Args:
        num_df: Claim filing indicator specific continuous columns dataframe.
        threshold: Variance threshold value. Default value is 0.1.
    
    Returns:
        num_df2: A copy of num_df but columns with low variance are dropped.
        dropped: List of columns that are dropped.
    
    """
    
    # Creating a Variance Threshold object
    var_thresh = VarianceThreshold(threshold)
    
    # Fitting the var_thresh on num_df
    var_thresh.fit(num_df)
    
    # Removing columns that have low variance
    num_df2 = num_df[num_df.columns[var_thresh.get_support(indices=True)]]
    
    # List of columns that are dropped
    dropped = np.setdiff1d(num_df.columns, num_df2.columns)
    
    return num_df2, dropped


def correlation(df, threshold):
    """ This method identifies correlation between columns in the DataFrame and removes one of the two correlated columns if the correlation value is greater than the threshold.
        
    Args:
        df: Claim filing indicator specific dataframe.
        threshold: Correlation threshold value.
    
    Returns:
        col_corr: Set of columns that are dropped.
        df: DataFrame with correlated columns removed.
        
    """
    # Set of all the names of deleted columns
    col_corr = set()
    
    # Creating a correlation matrix from the dataframe
    corr_matrix = df.corr()
    
    # Iterating over each pair of columns to see if their correlation is greater than the threshold value.
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                print("Correlated columns are {0} and {1} with value: {2}".format(corr_matrix.columns[i],corr_matrix.columns[j],abs(corr_matrix.iloc[i, j])))
                print('*****************\n')
                col_corr.add(colname)
                if colname in df.columns:
                    if colname!= "total_claim_charge_amount":
                        print(colname)
                        # deleting the column from the dataset
                        del df[colname]
    return col_corr, df


def filter_numeric_cols(num_df):
    """ Performs Feature Selection on Continuous Features based on Variance, Correlation etc.
    
    Args:
        num_df: Claim filing indcator specific continuous features dataframe.
    
    Returns:
        num_stats: .describe output of the retained continuous columns that satisfies all the condition checks.
        filtered_cat_columns_tracker: A tracker dictionary that contains the condition name and a list of columns that are dropped due to that condition.
        
    """
    
    print("***Begin filter_numeric_cols method***")
    
    # Dictionary to track filter conditions and the corresponding columns that are removed.
    filtered_num_columns_tracker = {}
    
    # Converting all columns into pandas "float" type
    num_df = num_df.astype("float")
    
    # Condition#1: Removing low variance columns
    num_df, dropped = variance_threshold_selector(num_df)
    filtered_num_columns_tracker["low_variance"] = list(dropped)
    
    # Condition#2: Removing columns that are "highly" correlated
    dropped, num_df= correlation(num_df, 0.80)
    filtered_num_columns_tracker["high_correlation"] = list(dropped)
    
    num_stats = num_df.describe().transpose()
    
    """Not Used"""
    # Condition#3: Removing columns based on "standard deviation"
    #mask = num_stats["std"]>0
    #filtered_num_columns_tracker["standard_deviation"] = list(num_stats[mask].index)
    #num_stats = num_stats[~mask]
    
    # Sorting num_stats based on "std" and "mean" in ascending order
    num_stats = num_stats.sort_values(["std", "mean"], ascending=False)
    
    print("***End filter_numeric_cols method***")
    print("***########################################***\n\n")
    #print(to_drop)
    return num_stats, filtered_num_columns_tracker


def get_dtype_dfs(df, claim_filing_ind_name, cat_num_filtered):
    """ This method takes the complete institutional or professional claims dataframe and returns continuous and categorical features in two separate dataframe for a specific claim filing indicator.
    
    Args:
        df: Complete Institutional and Professional claims dataframe.
        claim_filing_ind_name: Claim filing indicator value. Examples: Medicare Part A, Medicaid etc.
        cat_num_filtered: A dictionary containing the claim filing indicator name as key and their corresponding continuous and categorical feature names in two separate lists.
    
    Returns:
        cat_df: Claim filing indicator specific categorical columns dataframe.
        num_df: Claim filing indicator specific continuous columns dataframe.
        
    """
    print("***Begin get_dtype_dfs method***")
    
    # Get the list of claim filing indicator specific categorical and continuous columns.
    cat_cols, num_cols = cat_num_filtered[claim_filing_ind_name][0], cat_num_filtered[claim_filing_ind_name][1]
    
    # Create the subset of categorical and continuous features dataframes.
    cat_df, num_df = df[cat_cols], df[num_cols]
    print("***End get_dtype_dfs***")
    return cat_df, num_df


def cvsplits_to_testfoldvector(cvsplits, nrows=None, check=True):
    """ Convert a list of pairs of sets of sequential row indices 
    (train and test CV folds' splits, e.g., as produced by split() method in sklearn's CV)
    to a vector of test fold ids, that would be suitable for the use as input to PredefinedSplit.
    Any row that has no index in cvsplits is assigned a fold id of -1.
    Assume that each index belongs to a single test fold.
    
    cvsplits: list of CV train-test index array pairs
    nrows: if given, and is larger than any index in cvsplits, 
    check: whether to perform consistency checks for cvsplits
    
    Example:
    x = [(array([1, 3, 4]), array([0])),
         (array([0, 1, 3]), array([4])),
         (array([0, 4]), array([1, 3]))]
    cvsplits_to_testfoldvector(x)"""
    
    if check:
        idset = set()
        for f in cvsplits:
            tr = set(f[0])
            ts = set(f[1])
            if len(tr & ts):
                raise Exception("Overlapping train and test indices in cvsplits")
            idfold = tr | ts
            if len(idset) and idset != idfold:
                raise Exception("Folds in cvsplits contain different indices")
            idset |= idfold
    test_folds = [f[1] for f in cvsplits]
    N = np.max([max(f) for f in test_folds]) + 1
    if nrows is not None:
        if nrows < N:
            raise Exception(f"nrows is less than {N}")
        if nrows > N:
            N = nrows
    fold_vec = np.zeros((N))
    fold_vec.fill(-1)
    for f in range(len(test_folds)):
        fold_vec[test_folds[f]] = f
    return fold_vec


def create_cv_folds(xft, n_splits, random_state, to_pickle=None, verbose=False):
    """ Stratified CV folds grouped by a claims """
    # also store some outcome and bookkeeping columns
    #xft['time_to_pay_bins']=pd.qcut(xft.responseTime, q=2,labels=[0,1])
    xft["time_to_response_bins"] = np.random.choice([0,1], xft.shape[0])
    y=xft["time_to_response_bins"].values
    y= y.astype("float")
    yRef= xft.response_time

    X = xft.copy()    
    X_cols = ["time_to_response_bins", "medical_record_number"]
    X = xft[X_cols]
    gsplits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    agg_func="mean", random_state=random_state)
    
    folds = [(train_index, test_index)
              for train_index, test_index
              in gsplits.split(X, y, X.medical_record_number)]

    test_fold_vec = cvsplits_to_testfoldvector(folds)
    X["fold"] = test_fold_vec
    
    if verbose:
        print("fold  #train  #test  timeToResponse   timeToResponseC   Tresponse_tst  Tresponse_shared")
        
        for i in range(10):
            re = yRef.iloc[folds[i][1]].mean()
            re1 = xft.iloc[folds[i][1]][xft.event_flag==True]["response_time"].mean()
            nclaim = xft.index[folds[i][1]].nunique()
            xclaim = len(set(X.index[folds[i][0]]) &
                       set(X.index[folds[i][1]])) # intersect must be empty
            print(f"{i} {len(folds[i][0])} {len(folds[i][1])} {re1:.5} {re:.5} {nclaim} {xclaim}")
            
    if to_pickle:
        pickl(to_pickle, (folds, gsplits, X))
    
    return folds, gsplits, X


def get_folds(df):
    """ This methods takes in the claim filing indicator specific dataframe and returns the k-folds indicaes for k-fold validation."""
    
    print("***Begin get_folds method***")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folds, gsplits, X = create_cv_folds(df, 10, 123,verbose=True)
    
    print("***End get_folds method***")
    
    return folds


def run_grid_search(df, rsf_df, grid_search_params, folds):
    """ This method runs a GridSearch with a 10-fold cross_validation. 
    
    Args:
        df: Claim filing indicator specific dataframe.
        rsf_df: DataFrame with final set of features that are preprocessed - Label Encoding etc.
        grid_search_params: List of grid search parameters to iterate over.
        folds: List of lists containing 10-folds train and test indices.
        
    Returns:
        Returns a list of train and test set c-index values.
        
    """
    
    # List to keep track of c-index values for a combination of grid search params, train and test c-index values as a tuple.
    c_index_values=[]
    
    
    # Iterating over the grid search parameters.
    for i in range(len(grid_search_params)):
        
        # Fit the RSF model with the params
        rsf = RandomSurvivalForest(n_estimators = grid_search_params[i][0],
                                   max_features = grid_search_params[i][1],
                                   max_depth = grid_search_params[i][2],
                                   min_samples_split = grid_search_params[i][3],
                                   min_samples_leaf = grid_search_params[i][4],
                                   n_jobs = -1,
                                   oob_score=True)
        
        # Iterate over the 10-folds to capture train and test c-index values.
        for j in range(len(folds)):
            print(f"In combo: {i} and fold: {j}")
            
            # Train Set
            X_train = rsf_df.iloc[folds[j][0]].reset_index(drop=True)
            y_train = Surv.from_arrays(np.repeat(True, len(X_train)), df.iloc[folds[j][0]]["response_time"].reset_index(drop=True))

            # Test Set
            X_test = rsf_df.iloc[folds[j][1]].reset_index(drop=True)
            y_test = Surv.from_arrays(np.repeat(True, len(X_test)), df.iloc[folds[j][1]]["response_time"].reset_index(drop=True))
            
            # Fitting the model
            rsf.fit(X_train, y_train)
            
            # Capture the train and test set c-index values.
            c_index_values.append((grid_search_params[i], j, rsf.score(X_train, y_train), rsf.score(X_test, y_test), rsf.oob_score_))
            
    return c_index_values


def ten_fold_validation(rsf_df, folds):
    c_index_values = []
    feature_importance_df = pd.DataFrame()
    
    y = Surv.from_arrays(np.repeat(True, len(rsf_df)), rsf_df["response_time"].values)
    
    sub_df = rsf_df.drop(columns=["response_time", "event_flag"])
    
    
    for i in range(10):
        X_train = sub_df.iloc[folds[i][0]].reset_index(drop=True)
        y_train = y[folds[i][0]]
        
        X_test = sub_df.iloc[folds[i][1]].reset_index(drop=True)
        y_test = y[folds[i][1]]
        
        rsf = RandomSurvivalForest(oob_score=True)

        rsf.fit(X_train, y_train)

        c_index_values.append((i, rsf.score(X_train, y_train), rsf.score(X_test, y_test), rsf.oob_score_))
        
        feature_names = list(X_test.columns)
        perm = permutation_importance(rsf, X_test, y_test, n_repeats=3)
        temp_df = pd.DataFrame({"feat_name":feature_names, "imp_score_"+str(i):perm.importances_mean})
              
        # Using "feature" column as index
        temp_df = temp_df.set_index("feat_name")
        feature_importance_df = pd.concat([feature_importance_df, temp_df], axis=1)
    
    concordance_df = pd.DataFrame(c_index_values, columns=["fold_num", "train_score", "test_score", "oob_score"])
    return feature_importance_df, concordance_df


def recursive_feature_elimination(folds, rsf_df, features_list):
    # Iterations tracker
    i = 0
    
    # Tracker how features are filtered
    features_tracker = {}
    
    # feature_importance_tracker
    feat_imp_tracker = {}
    
    # c-index tracker
    c_index_tracker = {}
    
    while len(features_list)!=0:
        print("Current Iteration#: {}".format(i))
        
        features_tracker[i] = features_list.copy()
        
        feature_importance_df, c_index_values_df = ten_fold_validation(rsf_df, folds)
        
        print(feature_importance_df)
        
        feat_imp_tracker[i] = feature_importance_df
        
        c_index_tracker[i] = {"mean": c_index_values_df[["train_score", "test_score", "oob_score"]].mean(), 
                              "std": c_index_values_df[["train_score", "test_score", "oob_score"]].std()}
        
        i += 1
        weight_cols = [column for column in feature_importance_df.columns if "imp_score_" in column]
        feat_imp = feature_importance_df[weight_cols].mean(axis=1).sort_values(ascending=False)
        
        feat_to_remove = feat_imp[feat_imp==feat_imp.min()].index[0]
        features_list.remove(feat_to_remove)
        
        rsf_df = rsf_df.drop(columns=[feat_to_remove])
    
    return features_tracker, c_index_tracker, feat_imp_tracker


def compute_mean_ibs(rsf, X, y):
    # Using model object to predict survival probabilities for X_test
    X_test_preds = rsf.predict_survival_function(X, return_array=False)
    
    # Retrieving minimum and maximum values of event times
    min_time = int(min(rsf.event_times_))
    max_time = int(max(rsf.event_times_))
    extend_time = 90

    # Forward fill the survival predictions
    Survival = [[fn(i) if i<=max_time else fn(max_time) for i in range(min_time, extend_time)] for fn in X_test_preds]
    
    # Adding survival predictions at 0th time till min_time
    Survival = [[1 for i in range(min_time)] + sub_list for sub_list in Survival]

    Survival = np.asarray(Survival)

    event_times = {index:int(value[1]) for index, value in enumerate(y)}

    times = np.asarray([i for i in range(90)])
    ibs = []

    for index, survival in enumerate(Survival):
        survival = survival.reshape(1, -1)
        a = (1-survival[0][:event_times[index]])**2
        b = (0-survival[0][event_times[index]:])**2
        brier_scores = np.concatenate((a, b))
        ibs.append(np.trapz(brier_scores, times)/90)

    return np.mean(ibs)


def _check_estimate_1d(estimate, test_time):
    estimate = check_array(estimate, ensure_2d=False)
    if estimate.ndim != 1:
        raise ValueError(
            'Expected 1D array, got {:d}D array instead:\narray={}.\n'.format(
                estimate.ndim, estimate))
    check_consistent_length(test_time, estimate)
    return estimate
    
    
def _check_inputs(event_indicator, event_time, estimate):
    check_consistent_length(event_indicator, event_time, estimate)
    event_indicator = check_array(event_indicator, ensure_2d=False)
    event_time = check_array(event_time, ensure_2d=False)
    estimate = _check_estimate_1d(estimate, event_time)

    if not np.issubdtype(event_indicator.dtype, np.bool_):
        raise ValueError(
            'only boolean arrays are supported as class labels for survival analysis, got {0}'.format(
                event_indicator.dtype))

    if len(event_time) < 2:
        raise ValueError("Need a minimum of two samples")

    if not event_indicator.any():
        raise ValueError("All samples are censored")

    return event_indicator, event_time, estimate
    

def _get_comparable(event_indicator, event_time, order):
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        start = i + 1
        end = start
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time
        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
        i = end

    return comparable, tied_time
    

def _estimate_concordance_index(event_indicator, event_time, estimate, weights, tied_tol=1e-8):
    order = np.argsort(event_time)

    comparable, tied_time = _get_comparable(event_indicator, event_time, order)

    if len(comparable) == 0:
        raise NoComparablePairException(
            "Data has no comparable pairs, cannot estimate concordance index.")

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        w_i = weights[order[ind]]

        est = estimate[order[mask]]

        assert event_i, 'got censored sample at index %d, but expected uncensored' % order[ind]

        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        # an event should have a higher score
        con = est < est_i
        n_con = con[~ties].sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * mask.sum()

        tied_risk += n_ties
        concordant += n_con
        discordant += est.size - n_con - n_ties

    cindex = numerator / denominator
    return cindex, concordant, discordant, tied_risk, tied_time


def compute_c_index(rsf, X, y):
    event_indicator = np.asarray([val[0] for val in y]) 
    event_time = np.asarray([val[1] for val in y]) 
    estimate = rsf.predict(X) 
    tied_tol=1e-8
    
    event_indicator, event_time, estimate = _check_inputs(event_indicator, event_time, estimate)

    w = np.ones_like(estimate)

    result = _estimate_concordance_index(event_indicator, event_time, estimate, w, tied_tol)
    return result[0]


