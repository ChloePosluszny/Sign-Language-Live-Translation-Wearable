import os
import pandas as pd

'''
Usage:
GetFiles() gets each CSV file name and groups them by word/letter or by user
ConcatenateFiles_Multiple() concatenates the CSVs in each group into one CSV for that group
ConcatenateFiles_All() gets all the CSVs and combines them into a single CSV file
'''

TRAINING_DATA_PATH = "training_data"
CONCATENATED_DATA_PATH = "concatenated_data"

GROUPING_MODE = 1 # 0 = no grouping (single csv), 1 = grouping (multiple csv)
SPLIT_GROUPING = 1 # 0 = group by title, 1 = group by subtitle

# Key is the grouping to be saved as one file
# Value is the name of each file for the group
training_files = {}

# Get each individual training data file (deprecated)
def GetFiles():
    for root, dirs, files in os.walk(f"{TRAINING_DATA_PATH}/"):
        for file in files:
            file_split = file.split("_")[SPLIT_GROUPING]
            if "." in file_split:
                file_split = file_split.split(".")[0]
            if file_split not in training_files:
                training_files[file_split] = []
            training_files[file_split].append(file)
    return

# Concatenate the individual training data files into concatenated training data files
def ConcatenateFiles_Multiple():
    for basefile in training_files:
        data = []
        for file in training_files[basefile]:
            df = pd.read_csv(f"{TRAINING_DATA_PATH}/{file}")
            data.append(df)
        new_Data = pd.concat(data)
        new_Data.to_csv(f"{CONCATENATED_DATA_PATH}/{basefile}.csv", index=False)
    return

# Concatenate the individual training data files into one training data file
def ConcatenateFiles_All():
    data = []
    for root, dirs, files in os.walk(f"{TRAINING_DATA_PATH}/"):
        for file in files:
            df = pd.read_csv(f"{TRAINING_DATA_PATH}/{file}")
            data.append(df)
    new_data = pd.concat(data)
    new_data.to_csv(f"database.csv", index=False)
    return

if GROUPING_MODE == 1:
    GetFiles()
    ConcatenateFiles_Multiple()
else:
    ConcatenateFiles_All()
