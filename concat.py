import os
import pandas as pd

INDIVIDUAL_TRAINING_DATA = "individual_training_data"
CONCATENATED_TRAINING_DATA = "concatenated_training_data"

# Key is the letter/word to be saved as one file
# Value is the name of each file for the letter/word
training_files = {}

# Get each individual training data file
def GetFiles():
    for root, dirs, files in os.walk(f"{INDIVIDUAL_TRAINING_DATA}/"):
        for file in files:
            file_split = file.split("_")
            if file_split[0] not in training_files:
                training_files[file_split[0]] = []
            training_files[file_split[0]].append(file)
    return

# Concatenate the individual training date files into concatenated training data files
def ConcatenateFiles():
    for basefile in training_files:
        data = []
        for file in training_files[basefile]:
            df = pd.read_csv(f"{INDIVIDUAL_TRAINING_DATA}/{file}")
            data.append(df)
        new_Data = pd.concat(data)
        new_Data.to_csv(f"{CONCATENATED_TRAINING_DATA}/{basefile}.csv", index=False)
    return

GetFiles()
ConcatenateFiles()
