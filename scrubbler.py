import csv
import os
import glob

# Define the fixed header
HEADER = [
    "hall_1", "hall_2", "hall_3",
    "flex_t", "flex_i", "flex_m", "flex_r", "flex_p",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "hand", "sign"
]

def scrub_csv(file_path):
    """
    Reads a CSV file at file_path, replaces any row with an incorrect number of columns
    by the next correct row, and writes the cleaned data back to the same file with a fixed header.
    """
    # Read all rows (including possibly bad header)
    with open(file_path, newline='') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    # Data rows start after the original header
    data_rows = rows[1:]
    num_cols = len(HEADER)
    
    scrubbed = []
    for idx, row in enumerate(data_rows):
        if len(row) != num_cols:
            # find next correct row
            replacement = None
            for next_row in data_rows[idx+1:]:
                if len(next_row) == num_cols:
                    replacement = next_row
                    break
            if replacement:
                scrubbed.append(replacement)
            # if no replacement found, skip this row
        else:
            scrubbed.append(row)

    # Write the scrubbed data back to the same file with the fixed header
    with open(file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(HEADER)
        writer.writerows(scrubbed)

def scrub_folder(folder_path):
    """
    Walks through all .csv files in folder_path, scrubs each one in place.
    """
    for filepath in glob.glob(os.path.join(folder_path, '*.csv')):
        scrub_csv(filepath)

# scrub_csv('training_data/w_r_R_dy.csv')
scrub_folder('training_data')