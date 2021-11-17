import json
import holidays
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import pickle

class CommonUtils():
    numeric_cols = []
    categorical_cols = []
    
    def __init__(self):
        return

    
    def append_to_numeric_cols(self, cols_list):
        """ Appends the column names in cols_list to numeric_cols variable.
        
        Args:
            cols_list: List of columns to be appended to the numeric_cols list.
        
        Returns:
            None
        """
        
        self.numeric_cols += cols_list
        return
    
    
    def append_to_categorical_cols(self, cols_list):
        """ Appends the column names in cols_list to categorical_cols variable.
        
        Args:
            cols_list: List of columns to be appended to the categorical_cols list.
        
        Returns:
            None
        """
        
        self.categorical_cols += cols_list
        return
    
    
    def get_numeric_cols(self):
        """ Returns the list of numerical columns names saved till the time of invoking this method.
        
        Args:
            None
        
        Returns:
            numeric_cols: List of numerical column names in the dataset.
        """
        
        return self.numeric_cols
    
    
    def get_categorical_cols(self):
        """ Returns the list of categorical columns names saved till the time of invoking this method.
        
        Args:
            None
        
        Returns:
            categorical_cols: List of categorical column names in the dataset.
        """
        
        return self.categorical_cols
    
    
    def save_cols_list(self, num_file_path, cat_file_path):
        """ Saves both numerical and categorical data column names as json files in the provided paths.
        
        Args:
            num_file_path: Path in which numerical data column names list needs to be saved.
            cat_file_path: Path in which categorical data column names list needs to be saved.
            
        Returns:
            None
        """
        with open(num_file_path, "w") as f:
            json.dump(list(set(self.numeric_cols)), f, indent=4)
        
        with open(cat_file_path, "w") as f:
            json.dump(list(set(self.categorical_cols)), f, indent=4)
        
        return
    
    
    def read_json(self, path, file_name):
        """Loading JSON files"""
        try:
            with open(path+file_name) as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error in reading file: {file_name}")
        return
        
    
    def read_csv(self, path, filename):
        """Loading CSVs"""
        try:
            with open(path+filename) as f:
                return pd.read_csv(f,encoding='utf-8')
        except Exception as e:
            raise Exception(f"Error in reading file: {filename}")
        return
    
    
    def add_prefix_strings(self, df, prefix_string, columns):
        """ Adds a prefix_string to all the values in the list of columns in a dataframe.
        
        Args:
            df: DataFrame to operate on.
            prefix_string: String value that needs to be prefixed.
            columns: List of column names in the DataFrame that needs to be modified.
            
        Returns:
            df: Modified DataFrame with Prefix String added values in the columns."""
        
        for column in columns:
            df[column] = df[column].apply(lambda x: prefix_string + "_".join(str(x).split(" ")))
        
        return df

    
    def get_count_of_lineitems(self, df, ref_cols):
        """ Counts the number of line items present in each claim file.
        
        Args:
            df: DataFrame containing the claims information.
            ref_cols: List of column names, used to uniquely identify a claim file in the dataset.
            
        Returns:
            df: DataFrame with an additional column, capturing the count of line items for each claim file."""
    
        count_df = df.groupby(ref_cols).size().reset_index(name="count_of_lineitems")
        df = df.merge(count_df, on=ref_cols, how="inner")
        self.append_to_numeric_cols(["count_of_lineitems"])
        
        return df


    def get_count_of_other_subscribers(self, df):
        """ Counts the number of Other Subscribers available in each claim file.
        
        Args:
            df: DataFrame containing the claims information.
            
        Returns:
            df: DataFrame with an additional column, capturing the count of other subscribers for each claim file.
        """
        
        regex_string = "OtherSubscriberInformation_SBR01-PayerResponsibilitySequenceNumberCode"
        
        cols_list = list(df.filter(regex=regex_string, axis=1).columns)
        
        df["count_of_other_subscribers"] = len(cols_list)-df[cols_list].isnull().sum(axis=1)
        
        self.append_to_numeric_cols(["count_of_other_subscribers"])
        
        return df
    
    
    def get_agg_features(self, df, ref_cols, param):
        """ Creates aggegate features such as sum, mean and median values based on Service Line Item Quantity and Service Line Item Quantity Units, for each unique units values available in the dataset.
        
        Args:
            df: DataFrame containing the claims information.
            ref_cols: List of column names, used to uniquely identify a claim file in the dataset.
            param: Placeholder argument to indicate whether the claim type is Institutional or Professional. Units "MJ" are available in Professional claims only and units "DA" are available in Institutional claims only.
            
        Returns:
            df: DataFrame with additional columns, capturing the sum, mean and median of Service Line Quantity Values for each claim file.
        
        """
        
        # Service Line Quantity and Service Line Quantity Units column names vary between Professional and Institutional Claims. Depending on the claim type, param value is used to retrieve the appropriate column names.
        if param=="MJ":
            # Professional
            qty_col = "Loop2400_SV1-Segment-ProfessionalService_SV104-Quantity-380"
            qty_units = "Loop2400_SV1-Segment-ProfessionalService_SV103-UnitorBasisforMeasurementCode-355"
            sub_str = "mins_"
        
        elif param=="DA":
            # Institutional
            qty_col = "Loop2400_SV2-Segment-InstitutionalServiceLine_SV205-Quantity-380"
            qty_units = "Loop2400_SV2-Segment-InstitutionalServiceLine_SV204-UnitorBasisforMeasurementCode-355"
            sub_str = "days_"
        
        # Converting Service Line Quantity values into Float type.
        df[qty_col] = df[qty_col].astype("float")
        
        # Pivoting based on ref_cols and applying aggregate functions on qty_col using qty_units values.
        # Filling the NaN/Null values with 0.
        temp_df = df.pivot_table(index=ref_cols, columns=qty_units, values=qty_col, aggfunc=["sum", "mean", "median"]).fillna(0)
        
        # Creating the column names in accordance with the qty_units values for consistency and better interpretability.
        # Units - "units" are common between Professional and Institutional claim types.
        temp_df.columns = ["units_"+agg_op+"_quantity" if "UN" in unit else sub_str+agg_op+"_quantity" for agg_op, unit in temp_df.columns]
        
        temp_df.reset_index(inplace=True)
        
        # Inner join on ref_cols between df and temp_df
        df = df.merge(temp_df, on=ref_cols, how="inner")
        
        # Appending the aggregate columns to numeric columns list
        agg_cols_list = list(df.filter(regex="_quantity", axis=1).columns)
        # agg_cols = [col for col in df.columns if "_quantity" in col]
        self.append_to_numeric_cols(agg_cols_list)
        
        return df


    def reduce_unique_values(self, df, column_name, new_column_name, percentage, replace_string):
        """ Groups all the categorical values, that have less than a defined percentage of data entries under column_name into Others group.
        
        Args:
            df: DataFrame containing the claims information.
            column_name: Name of the column in which cardinality needs to be reduced.
            new_column_name: New column name into which the cardinality reduced data is written into.
            percentage: Percentage value, any category with percentage of rows lesser than this are grouped into others.
            replace_string: String value of "Others" bucket.
            
        Returns:
            df: DataFrame with additional column, capturing the reduced cardinality values in the dataset.
        """
        
        claim_filing_ind_column = "Loop2000B_SBR-Segment-SubscriberInformation_SBR09-ClaimFilingIndicatorCode-1032"
        
        df[new_column_name] = ""
        
        for claim_filing_ind_code in df[claim_filing_ind_column].unique():
            mask = df[claim_filing_ind_column]==claim_filing_ind_code
            temp_vc = df.loc[mask, column_name].value_counts()
            replace_list = list(temp_vc[((temp_vc/temp_vc.sum())*100).lt(1)].index)
            df.loc[mask, new_column_name] = df.loc[mask, column_name].apply(lambda x: replace_string if x in replace_list else x)
        
        return df


    def binarize_a_series(self, series):
        """ Based on mask condition, a series of values are binarized - Yes/No.
        Used especially for converting CPT, HCPCS and NDC code values at Line level to Claim level.
        
        Args:
            series: Series of values to binarize based on a condition.
        
        Returns:
            series: Updated Series with binary values - Yes/No.
        
        """
        
        mask = series>0
        
        series[mask] = "Yes"
        
        series[~mask] = "No"
        
        return series


    def serviceline_to_claimlevel(self, df, ref_cols, column_name):
        """ Converts Service Line Item data such as CPT, HCPCS and NDC codes values into One Hot Encoded Vectors (Yes/No) at Claim Level.
        
        Args:
            df: DataFrame containing the claims information.
            ref_cols: List of column names, used to uniquely identify a claim file in the dataset.
            column_name: Name of column, whose values are converted into One Hot Encoded Vectors at Claim Level.
        
        Returns:
            df: DataFrame with additional columns - one hot encoded vector columns (Yes/No).
        """
        
        temp_df = df[ref_cols+[column_name]].copy()
        temp_df["val"] = 1
        temp_df = temp_df.fillna("nan_value")
        temp_df = temp_df.pivot_table(index=ref_cols, columns=column_name, values="val", aggfunc="sum")
        temp_df = temp_df.fillna(0)
        temp_df[temp_df.columns] = temp_df[temp_df.columns].apply(self.binarize_a_series)
        temp_df = temp_df.reset_index()
        df = df.merge(temp_df, on=ref_cols, how="inner")
        
        return df 


    def get_claim_level_data(self, df, cols_list):
        """ Drops all the duplicate rows, to retrieve unique claims dataset.
        
        Args:
            df: DataFrame containing the claims information.
            cols_list: List of column names, used for dropping duplicate rows.
            
        Returns:
            df: DataFrame containing unique claims information.
        
        """
        df = df[cols_list].drop_duplicates(keep="first").reset_index(drop=True)
        return df


    def has_multiple_claimfilingindicators(self, x):
        """ Helper method for correct_payer_claimfiling_combinations method.
        Identifies if a payer name is associated with multiple claim filing indicator code values.
        
        Args:
            x: Series of values.
        
        Returns:
            A Boolean value depending on the number of claim filing indicator code values. 
            Returns True if the payer has more than one unique claim filing indicator code values.
            Returns False if a payer has only one unique claim filing indicator code.
        """
        
        if x.notnull().sum() > 1:
            return True
        
        return False


    def correct_payer_claimfiling_combinations(self, df):
        """ Corrects the payer with more than one unique claim filing indicator code values.
        Assigns the payer to the claim filing indicator code based on the maximum number of claims.
        
        Args:
            df: DataFrame containing the claims information.
            
        Returns:
            df: DataFrame with additional column, corrected claim filing indicator code column.
        """
        
        columns = ["payer_name", "claim_filing_ind_code"]
        temp_df = df[columns].groupby(columns).size().reset_index(name="count")
        temp_df = temp_df.pivot_table(index="payer_name", columns="claim_filing_ind_code")
        temp_df.columns = [j for i, j in temp_df.columns]
        correction_mask = temp_df.apply(lambda x: self.has_multiple_claimfilingindicators(x), axis=1)
        temp_df = temp_df[correction_mask]
        temp_df = temp_df.idxmax(axis=1)

        payer_dict = dict(zip(list(temp_df.index), list(temp_df.values)))

        df["claim_filing_ind_code2"] = df["claim_filing_ind_code"].copy()
        for payer_name in payer_dict.keys():
            df.loc[df["payer_name"]==payer_name, "claim_filing_ind_code2"] = payer_dict[payer_name]

        return df


    def merge(self, left_df, right_df, merge_string, how, suffixes, target_column_name):
        """ Helper method for code_mapping method- for converting code values such as ICD-10-CM, ICD-10-PCS, Condition Codes etc., into their respective Heirarchical groups or Description using joins.
        
        Args:
            left_df: Claims Dataframe.
            right_df: Varies depending on the current column in context - either CCSR codes or CCS codes or Condition codes description etc.
            merge_string: Column name to merge left and right dfs.
            how: Captures the type of join - inner, left or right.
            suffixes: String to append incase if left_df and right_df common column names.
            target_column_name: String value to rename the merge_string column name for later convenience.
        
        Returns:
            left_df: Updated dataframe with additional columns.
        """
        
        right_df = right_df.rename(columns={merge_string:target_column_name})
        left_df = left_df.merge(right_df, on=target_column_name, how=how, suffixes=suffixes)
        right_columns = [col for col in right_df.columns if col!=target_column_name]
        columns = {k:k+'_'+target_column_name for k in left_df.keys() if k in right_columns}
        left_df.rename(columns=columns, inplace=True)
        
        return left_df


    def code_mapping(self, df, right_df, col_string, merge_string, right_string):
        """Mapping the ICD-10 code values to their description"""
        col_list = list(df.filter(regex=col_string, axis=1).columns)
        new_col_list = [col_string+'_'+str(i+1) for i in range(len(col_list))]
        columns = {col_list[i]: new_col_list[i] for i in range(len(col_list))}
        df.rename(columns=columns, inplace=True)
        for col in new_col_list:
            df = self.merge(df, right_df, merge_string, "left", (False, False), col)
        df.drop(columns=new_col_list, inplace=True)
        new_col_list = list(df.filter(regex=right_string+col_string, axis=1).columns)
        
        df["count_of_"+col_string] = len(new_col_list)-df[new_col_list].isnull().sum(axis=1)
        
        df["count_of_unique_"+col_string] = df[new_col_list].nunique(axis=1)
        
        df["list_of_"+col_string] = df[new_col_list].apply(lambda row: [x for x in row.values.tolist() if x==x], axis=1)
        
        return df
    
    
    def convert_codes(self, df, col_name, convert_dict):
        """Converting Codes to their descriptions"""
        df[col_name] = df[col_name].replace(to_replace = convert_dict)
        self.append_to_categorical_cols([col_name])
        return df
    
    
    def fill_missing_values(self, df, src_col, dest_col):
        """Filling the missing facility name with the billing provider name"""
        mask = df[dest_col].isna()
        df.loc[mask, dest_col]=df.loc[mask, src_col]
        self.append_to_categorical_cols([dest_col])
        return df
    
    
    def compare_columns(self, df, col1, col2, new_col):
        """Comparing if the 2 columns are same or not"""
        mask = df[col1]==df[col2]
        df[new_col] = "No"
        df.loc[mask, new_col] = "Yes"
        self.append_to_categorical_cols([col1, col2, new_col])
        return df 


    def merge_two_cols(self, df, col1, col2, new_col):
        """ Extracts Patient data from Patient and Subscriber columns into a single column.
        
        Args:
            df: DataFrame containing the claims information.
            col1: Subscriber Data Column.
        """
        
        df[new_col] = df[col2]
        
        mask = df["ind_reltn_code"]=="Self"
        
        df.loc[mask, new_col] = df.loc[mask, col1]
        #self.append_to_categorical_cols([new_col])
        return df


    def compute_patient_age(self, df, sub_dob, pat_dob, claim_creation_date):
        """Calculating the age of the patient"""
        df[sub_dob] = pd.to_datetime(df[sub_dob])
        df[pat_dob] = pd.to_datetime(df[pat_dob])
        df[claim_creation_date] = pd.to_datetime(df[claim_creation_date])
        
        mask = df["subscriber_pat_reltn"]=="Self"
        df.loc[mask, pat_dob] = df.loc[mask, sub_dob]
        df["patient_age"] = df[claim_creation_date]-df[pat_dob]
        df["patient_age"] = round(df["patient_age"].dt.days/365, 2)
        return df


    def create_flag_column(self, df, col, new_col):
        df[new_col] = "No"
        mask = ~df[col].isna()
        df.loc[mask, new_col] = "Yes"
        return df

    
    def get_date_features(self, df):
        """Creating features from claim creation date"""
        df["claim_creation_weekday"] = df["claim_creation_date"].dt.weekday
        df["claim_creation_month"] = df["claim_creation_date"].dt.month
        df["claim_creation_dayofmonth"] = df["claim_creation_date"].dt.day
        df["claim_creation_quarter"] = df["claim_creation_date"].dt.quarter
        df["claim_creation_hour"] = df["claim_creation_time"].apply(lambda x: int(x[:2]))
        self.append_to_categorical_cols(["claim_creation_weekday", 
                                         "claim_creation_month", 
                                         "claim_creation_dayofmonth", 
                                         "claim_creation_quarter", 
                                         "claim_creation_hour"])
        return df
    
    
    def get_min_max_years(self, df):
        """Finding the min & max years from the data"""
        temp = df['claim_creation_date'].dt.year
        min_year, max_year = temp.min(), temp.max()
        years = [int(i) for i in range(min_year, max_year+2)]
        return years
    
    
    def get_holidays_list(self, years):
        """Getting the holiday list"""
        holidays_list = []
        for year in years:
            holidays_list.extend([ptr[0] for ptr in holidays.US(years = year).items()])
        return holidays_list
    
    
    def create_holiday_count_variable(self,years, holidays_list, claim_creation_date):
        """Creating the count of holiday feature"""
        start_date = claim_creation_date.date()
        fourteenth_date = list(pd.date_range(start_date, periods=2, freq='14D'))[1].date()
        def next_working_date(ideal_response_date):
            if ideal_response_date in holidays_list or ideal_response_date.weekday() in [5, 6]:
                ideal_response_date = list(pd.date_range(ideal_response_date, periods=2, freq='1D'))[1].date()
                return next_working_date(ideal_response_date)
            else:
                return ideal_response_date
        ideal_response_date=next_working_date(fourteenth_date)
        holidays_count = (pd.to_datetime(ideal_response_date)-pd.to_datetime(fourteenth_date)).days
        return holidays_count


