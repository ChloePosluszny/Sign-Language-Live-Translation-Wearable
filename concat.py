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

SEQUENCE_LENGTH = 20 # Amount to group data by for the RNN

# Key is the grouping to be saved as one file
# Value is the name of each file for the group
training_files = {}

# Align the data in the file to a multiple of SEQUENCE_LENGTH
def AlignFile(df):
    nbr_of_rows = len(df)
    rows_remaining = (nbr_of_rows) % SEQUENCE_LENGTH
    while rows_remaining:
        df = df.drop(df.index[-1])
        rows_remaining -= 1
    return df

# Get each individual training data file (deprecated)
def GetFiles():
    for root, dirs, files in os.walk(f"{TRAINING_DATA_PATH}/"):
        for file in files:
            # Expecting format like 'sign_trainer_glove.csv'
            parts = file.split("_")
        
            trainer = parts[1]
            glove = parts[2].split(".")[0]  # Remove ".csv"
            group_key = f"{trainer}_{glove.lower()}"  # e.g., "r_L"
            if group_key not in training_files:
                training_files[group_key] = []
            training_files[group_key].append(file)

# Concatenate the individual training data files into one training data file
def ConcatenateFiles_Multiple():
    for basefile in training_files:
        data = []
        for file in training_files[basefile]:
            print(file)
            df = pd.read_csv(f"{TRAINING_DATA_PATH}/{file}")

            # Check if it's a static sign (no '_dy' in filename)
            if "_dy" not in file:
                # Pad the static sign by repeating each row SEQUENCE_LENGTH times
                padded_rows = []
                for _, row in df.iterrows():
                    repeated = pd.DataFrame([row.values] * SEQUENCE_LENGTH, columns=df.columns)
                    padded_rows.append(repeated)
                df = pd.concat(padded_rows, ignore_index=True)
            else:
                # Align dynamic signs to SEQUENCE_LENGTH
                df = AlignFile(df)

            data.append(df)
        new_Data = pd.concat(data)
        new_Data.to_csv(f"{CONCATENATED_DATA_PATH}/{basefile}.csv", index=False)

GetFiles()
ConcatenateFiles_Multiple()
