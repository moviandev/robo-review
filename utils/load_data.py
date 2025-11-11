from datasets import load_dataset
import pandas as pd

def load_dataset_as_dataframe():
  # Load Dataset
  ds1 = load_dataset("moviandev/file-1-project-4", data_files="1429_1.csv")["train"]
  ds2 = load_dataset("moviandev/file-1-project-4", data_files="Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")["train"]
  ds3 = load_dataset("moviandev/file-1-project-4", data_files="Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")["train"]

  # Transform it in Dataframes
  df1 = ds1.to_pandas()
  df2 = ds2.to_pandas()
  df3 = ds3.to_pandas()

  # Return the 3 dataframes
  return df1, df2, df3

