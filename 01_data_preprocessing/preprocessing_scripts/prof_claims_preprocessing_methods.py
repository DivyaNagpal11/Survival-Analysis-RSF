import json
import pickle
import logging
import warnings
import holidays
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


warnings.filterwarnings("ignore")

from common_utility import CommonUtils

class ProfClaimsPreprocessing(CommonUtils):
    
    def __init__(self):
        return
              
    def get_days_taken_for_claim_filing(self, claims_data, ref_cols):
        """ This method creates the days taken for claim filing feature for Professional claims.
        
        Args:
            claims_data: Professional Claims Data
            ref_cols: List of reference columns to uniquely identify a claim file.
            
        Returns:
            claims_data: Professional Claims Data with days_taken_for_claim_filing feature added.
        
        """
        # Creating two columns- service_min_date and service_max_date columns with placeholder data.
        claims_data["service_min_date"] = claims_data["Loop2400_DTP-Segment-DateServiceDate_DTP03-DateTimePeriod-1251"]
        claims_data["service_max_date"] = claims_data["Loop2400_DTP-Segment-DateServiceDate_DTP03-DateTimePeriod-1251"]
        
        # Date Qualifier and Date Values.
        srv_date_qual = "Loop2400_DTP-Segment-DateServiceDate_DTP02-DateTimePeriodFormatQualifier-1250"
        srv_date = "Loop2400_DTP-Segment-DateServiceDate_DTP03-DateTimePeriod-1251"
        
        # Date Range Mask.
        mask = claims_data[srv_date_qual]=="RD8"
        
        # Splitting the date ranges using hyphen and taking the left hand side part as min date and right hand side part as max date.
        claims_data.loc[mask, "service_min_date"] = claims_data.loc[mask, srv_date].str.split("-", expand=True)[0]
        claims_data.loc[mask, "service_max_date"] = claims_data.loc[mask, srv_date].str.split("-", expand=True)[1]
        
        # Converting min and max dates into pandas datetime.
        claims_data["service_min_date"] = pd.to_datetime(claims_data["service_min_date"])
        claims_data["service_max_date"] = pd.to_datetime(claims_data["service_max_date"])
        
        # Using the service max date and claim creation date to create days_taken_for_claim_filing feature.
        claims_data["days_taken_for_claim_filing"] = (pd.to_datetime(claims_data["Segment_BHT-Segment-BeginningOfHierarchicalTransaction_BHT04-Date-373"]) - claims_data["service_max_date"]).dt.days
        
        # Considering most recent or min value of days_taken_for_claim_filing...
        claims_filing_days = claims_data[ref_cols+["days_taken_for_claim_filing"]].groupby(ref_cols).min()
        
        # Merging claims_data and claims_filing_days.
        claims_data = claims_data.merge(claims_filing_days, on=ref_cols, how="inner")
        
        claims_data.rename(columns={"days_taken_for_claim_filing_y":"days_taken_for_claim_filing"}, inplace=True)
        
        # Appending days_taken_for_claim_filing to numeric columns list
        self.append_to_numeric_cols(["days_taken_for_claim_filing"])
        
        return claims_data
    
      
    def get_LOS(self, claims_data, ref_cols):
        """ This method computes the duration in days between min and max service dates on the claim file.
        
        Args:
            claims_data: Professional Claims Data.
            ref_cols: List of reference columns to uniquely identify a claim file.
            
        Returns:
            claims_data: Professional Claims Data with LOS feature added.
        
        """
        # Min and Max service dates on a claim file.
        min_sd = claims_data[ref_cols + ["service_min_date"]].groupby(ref_cols).min()
        max_sd = claims_data[ref_cols + ["service_max_date"]].groupby(ref_cols).max()
        
        # Merging min_sd and max_sd on ref_cols.
        diff = min_sd.merge(max_sd, on=ref_cols, how="inner")
        
        # Creating LOS feature.
        diff["LOS"] = (pd.to_datetime(diff["service_max_date"]) - pd.to_datetime(diff["service_min_date"])).dt.days
        
        # Merging diff and claims_data on ref_cols.
        claims_data = claims_data.merge(diff, on=ref_cols, how="inner")
        
        # Appending LOS to numeric columns list
        self.append_to_numeric_cols(["LOS"])
        
        return claims_data