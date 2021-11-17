import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance


class BootstrapFeatImp:
    def __init__(self, num_df, cat_df, y, cv_folds):
        # Numerical Features DataFrame
        self.num_df = num_df
        
        # Categorical Features DataFrame
        self.cat_df = cat_df
        
        # Target Variable
        self.y = y
        
        # RandomSurvivalForest Parameters Dictionary
        #self.params = params
        # Decided to proceed with default params for feat imp
        
        # List of Cross Validation Folds - 500 folds
        self.cv_folds = cv_folds

        
        
    def bootstrap_num_feat_imp(self, cv_fold):
        """Fits a RandomSurvivalForest model on Numerical Features
        for the given Train and Test folds in cv_fold,
        computes the feature importance using Permutation Combination approach using test set and
        returns the feature names and importance scores in a dictionary.
        
        Args:
            cv_fold: List of Train and Test set fold indices lists.
        
        Returns:
            num_imp_dict: Dictionary containing numerical feature names and feature importance scores.
        
        """
        
        # Create Train and Test sets using cv_fold infomration from numerical features dataframe.
        X_train = self.num_df.loc[cv_fold[0]].reset_index(drop=True)
        y_train = self.y[cv_fold[0]]
        
        X_test = self.num_df.loc[cv_fold[1]].reset_index(drop=True)
        y_test = self.y[cv_fold[1]]
        
        # Create and Fit the RandomSurvivalForest model with default arguments
        rsf = RandomSurvivalForest(oob_score=True)

        rsf.fit(X_train, y_train)
        
        # Compute Feature Importance scores using PermutationCombination approach using rsf object and test sets.
        # We're going with a repeat value of 10. This is hardcoded.
        r = permutation_importance(rsf, X_test, y_test, n_repeats=10)
        
        # Retrieve the Feature Importance scores by taking mean values across the 10 repetitions.
        # Create a dictionary using numerical feature names and importance scores.
        num_imp_dict = dict(zip(self.num_df.columns, r.importances_mean))
    
        return num_imp_dict
    
    
    def run_numfeat_wrapper(self, cpu_count=mp.cpu_count()):
        """ Wrapper method for computing feature importance scores for numerical features.
        Uses multiprocessing package to utilize predifined cpu cores for computing feature importance scores across 500 cross-validation folds.
        Aggregates the feature importance results across 500 folds and creates another dataframe capturing feature names and the number of times the feature was ranked higher than the noise column.
        
        Args:
            cpu_count: Number of cores to be utilized as part of parallel processing.
            
        Returns:
            numfeat_df: DataFrame containing feature names and the feature importance scores across 500 folds.
            numfeat_noise_df: DataFrame containing feature names and the number of times a feature was ranked higher than noise column.
            
        """
        
        # Initializing the pool object for multiprocessing.
        # By default all the available CPU cores will be utilized for parallel processing.
        pool = mp.Pool(cpu_count)
        
        # List to store the feature importance dictionary returned by bootstrap_cat_feat_imp method across 500 folds.
        numfeat_results = []
        
        # This code is specifically written to show the progress bar using tqdm with multiprocess.
        # This is based on the blog post: https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        for result in tqdm(pool.imap(func=self.bootstrap_num_feat_imp, iterable=self.cv_folds), total=len(self.cv_folds)):
            numfeat_results.append(result)
        
        # Closing the pool object.
        pool.close()
        
        # Converting the list of dictionaries of feature importance scores into a dataframe.
        numfeat_df = pd.DataFrame(numfeat_results)
        
        # Initializing an empty dataframe to capture feature name and 
        # the count value indicating the number of times the feature was ranked above noise.
        numfeat_noise_df = pd.DataFrame(columns=["feature_name", "ge_noise"])
        
        # Loop variable
        i=0
        
        # Iterate through catfeat_df and populate the numfeat_noise_df.
        for col in numfeat_df.columns:
            ge_noise = (numfeat_df[col]>numfeat_df["noise"]).sum()
            numfeat_noise_df.loc[i] = [col, ge_noise]
            i+=1
        
        # Sorting the numfeat_noise_df based on the decreasing values of ge_noise (count greater than noise).
        numfeat_noise_df = numfeat_noise_df.sort_values(["ge_noise"], ascending=[False])
        
        return numfeat_df, numfeat_noise_df
    
    
    
    def bootstrap_cat_feat_imp(self, cv_fold):
        """Fits a RandomSurvivalForest model on Categorical Features
        for the given Train and Test folds in cv_fold,
        computes the feature importance using Permutation Combination approach using test set and
        returns the feature names and importance scores in a dictionary.
        
        Args:
            cv_fold: List of Train and Test set fold indices lists.
        
        Returns:
            cat_imp_dict: Dictionary containing categorical feature names and feature importance scores.
        
        """
        
        # Create Train and Test sets using cv_fold infomration from numerical features dataframe.
        X_train = self.cat_df.loc[cv_fold[0]].reset_index(drop=True)
        y_train = self.y[cv_fold[0]]
        
        X_test = self.cat_df.loc[cv_fold[1]].reset_index(drop=True)
        y_test = self.y[cv_fold[1]]
        
        # Create and Fit the RandomSurvivalForest model using params dictionary.
        rsf = RandomSurvivalForest(oob_score=True)

        rsf.fit(X_train, y_train)
        
        # Compute Feature Importance scores using PermutationCombination approach using rsf object and test sets.
        # We're going with a repeat value of 10. This is hardcoded.
        r = permutation_importance(rsf, X_test, y_test, n_repeats=10)
        
        # Retrieve the Feature Importance scores by taking mean values across the 10 repetitions.
        # Create a dictionary using numerical feature names and importance scores.
        cat_imp_dict = dict(zip(self.cat_df.columns, r.importances_mean))
        
        return cat_imp_dict
    

    
    def run_catfeat_wrapper(self, cpu_count=mp.cpu_count()):
        """ Wrapper method for computing feature importance scores for categorical features.
        Uses multiprocessing package to utilize predifined cpu cores for computing feature importance scores across 500 cross-validation folds.
        Aggregates the feature importance results across 500 folds and creates another dataframe capturing feature names and the number of times the feature was ranked higher than the noise column.
        
        Args:
            cpu_count: Number of cores to be utilized as part of parallel processing.
            
        Returns:
            catfeat_df: DataFrame containing feature names and the feature importance scores across 500 folds.
            catfeat_noise_df: DataFrame containing feature names and the number of times a feature was ranked higher than noise column.
         
        """
        # Initializing the pool object for multiprocessing.
        # By default all the available CPU cores will be utilized for parallel processing.
        pool = mp.Pool(cpu_count)
        
        # List to store the feature importance dictionary returned by bootstrap_cat_feat_imp method across 500 folds.
        catfeat_results = []
        
        # This code is specifically written to show the progress bar using tqdm with multiprocess.
        # This is based on the blog post: https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
        for result in tqdm(pool.imap(func=self.bootstrap_cat_feat_imp, iterable=self.cv_folds), total=len(self.cv_folds)):
            catfeat_results.append(result)
        
        # Closing the pool object.
        pool.close()
        
        # Converting the list of dictionaries of feature importance scores into a dataframe.
        catfeat_df = pd.DataFrame(catfeat_results)
        
        # Initializing an empty dataframe to capture feature name and 
        # the count value indicating the number of times the feature was ranked above noise.
        catfeat_noise_df = pd.DataFrame(columns=["feature_name", "ge_noise"])
        
        # Loop variable
        i=0
        
        # Iterate through catfeat_df and populate the catfeat_noise_df.
        for col in catfeat_df.columns:
            ge_noise = (catfeat_df[col]>catfeat_df["noise"]).sum()
            catfeat_noise_df.loc[i] = [col, ge_noise]
            i+=1
        
        # Sorting the catfeat_noise_df based on the decreasing values of ge_noise (count greater than noise).
        catfeat_noise_df = catfeat_noise_df.sort_values(["ge_noise"], ascending=[False])
        
        return catfeat_df, catfeat_noise_df