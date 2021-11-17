import json
import holidays
import pandas as pd

from common_utility import CommonUtils

class InstClaimsPreprocessing(CommonUtils):
    
    def __init__(self):
        return
    

    def get_count_of_invest_dev_exemp(self, df):
        regex_string = "InvestigationalDeviceExemptionNumber_REF02"
        cols_list = list(df.filter(regex=regex_string, axis=1).columns)
        df["count_of_invest_dev_exemp"] = len(cols_list)-df[cols_list].isnull().sum(axis=1)
        self.append_to_numeric_cols(["count_of_invest_dev_exemp"])
        return df

    
    def count_dx_codes(self, df):
        df["count_of_total_dx_codes"] = df[["count_of_PrincipalDiagnosis",
                                            "count_of_AdmittingDiagnosis",
                                            "count_of_Patient’SReasonForVisit",
                                            "count_of_ExternalCauseOfInjury_HI",
                                            "count_of_OtherDiagnosisInformation_HI"]].sum(axis=1)


        df["count_of_unique_total_dx_codes"] = df[["count_of_unique_PrincipalDiagnosis",
                                                   "count_of_unique_AdmittingDiagnosis",
                                                   "count_of_unique_Patient’SReasonForVisit",
                                                   "count_of_unique_ExternalCauseOfInjury_HI",
                                                   "count_of_unique_OtherDiagnosisInformation_HI"]].sum(axis=1)


        df["list_of_total_dx_codes"] = df[["list_of_PrincipalDiagnosis",
                                           "list_of_AdmittingDiagnosis",
                                           "list_of_Patient’SReasonForVisit",
                                           "list_of_ExternalCauseOfInjury_HI",
                                           "list_of_OtherDiagnosisInformation_HI"]].sum(axis=1)


        df["count_of_total_pcs_codes"] = df[["count_of_PrincipalProcedureInformation",
                                             "count_of_OtherProcedureInformation_HI"]].sum(axis=1)


        df["count_of_unique_total_pcs_codes"] = df[["count_of_unique_PrincipalProcedureInformation",
                                             "count_of_unique_OtherProcedureInformation_HI"]].sum(axis=1)


        df["list_of_total_pcs_codes"] = df[["list_of_PrincipalProcedureInformation",
                                            "list_of_OtherProcedureInformation_HI"]].sum(axis=1)
        
        count_columns = list(df.filter(regex=r"count_of", axis=1).columns)
        self.append_to_numeric_cols(count_columns)
        return df


    def newborn_codes(self, df, newborn_codes, admission_type_codes, admission_source_codes):
        mask = df["admission_type_code"]=="4"
        df.loc[mask, "admission_source_code"] = df.loc[mask, "admission_source_code"].replace(to_replace=newborn_codes)
        df = self.convert_codes(df, "admission_type_code", admission_type_codes)
        df = self.convert_codes(df, "admission_source_code", admission_source_codes)
        
        self.append_to_categorical_cols(["admission_source_code"])
        return df


    def data_corrections(self, df, col_name, correction_dictionary):
        #Using .replace method of pandas to substitute the misspelt ones with correct spellings. 
        df[col_name] = df[col_name].replace(to_replace=correction_dictionary)    
        return df


    def get_time_variables(self, df, statement_dates):
        df[["statement_from", "statement_to"]] = df[statement_dates].str.split("-",expand=True) 
        df["statement_from"] = pd.to_datetime(df["statement_from"])
        df["statement_to"] = pd.to_datetime(df["statement_to"])
        df["claim_creation_date"] = pd.to_datetime(df["claim_creation_date"])
        df["LOS"] = (df["statement_to"]-df["statement_from"]).dt.days
        df["days_taken_for_claim_filing"] = 0
        df["days_taken_for_claim_filing"] = (df["claim_creation_date"] - df["statement_to"]).dt.days
        
        self.append_to_numeric_cols(["LOS", "days_taken_for_claim_filing"])
        return df


