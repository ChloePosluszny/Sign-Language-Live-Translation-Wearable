import os
import pandas as pd

'''
Usage:
GetFiles() gets each CSV file name and groups them by word/letter
ConcatenateFiles() concatenates the CSVs in each word/letter group into one CSV for that word/letter group
CombineFiles() gets all the CSVs and combines them into a single CSV file
'''

TRAINING_DATA = "training_data"

# Key is the letter/word to be saved as one file
# Value is the name of each file for the letter/word
training_files = {}

# Get each individual training data file (deprecated)
def GetFiles():
    for root, dirs, files in os.walk(f"{TRAINING_DATA}/"):
        for file in files:
            file_split = file.split("_")
            if file_split[0] not in training_files:
                training_files[file_split[0]] = []
            training_files[file_split[0]].append(file)
    return

# Concatenate the individual training data files into concatenated training data files (deprecated)
def ConcatenateFiles():
    for basefile in training_files:
        data = []
        for file in training_files[basefile]:
            df = pd.read_csv(f"{TRAINING_DATA}/{file}")
            data.append(df)
        new_Data = pd.concat(data)
        new_Data.to_csv(f"{TRAINING_DATA}/{basefile}.csv", index=False)
    return

# Concatenate the individual training data files into one training data file (deprecated)
def CombineFiles():
    data = []
    for root, dirs, files in os.walk(f"{TRAINING_DATA}/"):
        for file in files:
            df = pd.read_csv(f"{TRAINING_DATA}/{file}")
            data.append(df)
    new_data = pd.concat(data)
    new_data.to_csv(f"database.csv", index=False)
    return

# GetFiles()
# ConcatenateFiles()
CombineFiles()
