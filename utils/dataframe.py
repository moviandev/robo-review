import pandas as pd
from utils.load_data import load_dataset_as_dataframe
from utils.data_cleaner import data_cleaner

def load_and_clean_data():
    # Load raw data as DataFrames
    df1, df2, df3 = load_dataset_as_dataframe()
    
    # List of DataFrames and their corresponding source names
    df_list = [df1, df2, df3]
    source_names = [
        "1429_1.csv",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv",
        "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    ]
    
    # Clean the data using the data_cleaner function
    cleaned_df = data_cleaner(df_list, source_names)
    
    return cleaned_df