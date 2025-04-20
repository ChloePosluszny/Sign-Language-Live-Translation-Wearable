import serial
import time
import csv
import torch  # PyTorch required for loading .pth models
import joblib # Required to load MLP Classifier from sklearn
import numpy as np
import os
import sklearn
from sklearn.neural_network import MLPClassifier
import pandas as pd
from Train_RNN import RNN
import warnings
warnings.filterwarnings('ignore')


trainer_names = {'ALFONSO': 'a1', 'CHLOE': 'c3', 'DAVID': 'd', 'ERIK': 'e', 'RAHMAN': 'r'}

# -------------------------------
# Global Mode Flags (toggle as needed)
# -------------------------------

TRAINING_MODE = False       # True: training (write to CSV), False: output (ML inference)
TRAINER_NAME = 'DAVID'

COMM_MODE = "SERIAL"       # Options: "BLUETOOTH" or "SERIAL"
GLOVE_MODE = "SINGLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)
HAND = "L"
# -------------------------------
# Communication Port Configuration
# -------------------------------
BLUETOOTH_COM_PORT = 'COM5'   # Port for Bluetooth
SERIAL_COM_PORT = 'COM3'      # Port for direct serial connection (e.g., USB)

# Select the appropriate COM port based on COMM_MODE
COM_PORT = BLUETOOTH_COM_PORT if COMM_MODE == "BLUETOOTH" else SERIAL_COM_PORT
BAUD_RATE = 115200           # Must match the ESP32's baud rate

# -------------------------------
# File, Model, and Sensor Settings
# -------------------------------
CSV_FILE_PATH = None  # Will be set based on user input if TRAINING_MODE is True
CSV_TITLE = None      # Global title for the CSV
CSV_SUBTITLE = None
iterations = 100
MODEL_PATH = f"RNN_model_{HAND}.pth"
model = None
scaler = None
label_encoder  = None
seq_len = 20
WORD = ''
feature_length = 15

# Predefined sensor names (modify as needed for your setup)
HEADER = ["hall_1", "hall_2", "hall_3", "flex_t", "flex_i", "flex_m", "flex_r", "flex_p", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "hand", "sign"]

def load_model():
    """
    Loads the PyTorch model from the specified MODEL_PATH.
    """
    global model
    global scaler
    global label_encoder
    global all_labels
    print(f"Attempting to load model from: {MODEL_PATH}")
    try:
        model = torch.load(MODEL_PATH, weights_only=False)
        model.eval()
        label_encoder = joblib.load(f"label_encoder_{HAND}.pkl")
        scaler = joblib.load(f"scaler_{HAND}.pkl")
  

        # model = joblib.load(MODEL_PATH)
        # scaler = joblib.load("scaler.pkl")
        print(f"{MODEL_PATH} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    
RNN_buffer = []
def translate_data():
    """
    Placeholder function for translating glove sensor data using a machine learning model.
    
    Args:
        poll_data: A list containing one or two lists of floats (data from glove(s)).
        
    Returns:
        A string representing the translated output.
    """
    global WORD
    if model is None:
        return "Model not loaded"
    # Example: Convert poll_data to a tensor, process it with the model, then decode the result.
    else:
       
        
        #poll data should be a list of lists of sequence length data entries
       
        normalized_buffer = scaler.transform(RNN_buffer)
        tensor_input = torch.tensor(normalized_buffer, dtype=torch.float32).unsqueeze(0)
        if tensor_input.size(-1) != feature_length:
            print(f"Expected 15 features, but got {tensor_input.size(-1)}")
            return "Invalid input shape"

        
        with torch.no_grad():
            outputs = model(tensor_input)
            probs = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probs, dim=1)

            all_labels = label_encoder.classes_
            prob_values = probs[0]

            label_prob_pairs = []
            for i in range(len(all_labels)):
                label_prob_pairs.append((all_labels[i], prob_values[i]))

            sorted_pairs = sorted(label_prob_pairs, key=lambda x: x[1], reverse=True)

            print("Labels and Probabilities:\n")
            for label, prob in sorted_pairs[:3]:
                print(f"Label: {label}, Probability: {prob.item() * 100:.2f}%")

            predicted_label = label_encoder.inverse_transform(predicted)
            # print(f"\nOutput: {predicted_label[0]} with confidence {max_prob.item() * 100:.2f}") 
            print(f"\n\033[32mOutput: {predicted_label[0]} with confidence {max_prob.item() * 100:.2f}%\033[0m")
            # WORD += str(predicted_label[0])


translations = []
def process_poll(poll_data):
    """
    Processes a complete poll of sensor data based on the current mode.
    
    For training, it combines the arrays (if DOUBLE mode, concatenates the two arrays)
    and appends them to a CSV file with the CSV title appended to both the header and each row.
    For output mode, it sends the data to the ML model.
    
    Args:
        poll_data: List of arrays from the glove(s).
    """
    # global old_letter
    global RNN_buffer
    if TRAINING_MODE:
        if GLOVE_MODE == "DOUBLE":
            combined_data = poll_data[0] + poll_data[1]
        else:  # SINGLE mode
            combined_data = poll_data[0]
        
        if len(combined_data) != feature_length:
            print("skipping")
            return False
        RNN_buffer.append(combined_data)
        # Write to CSV with the CSV_TITLE appended.
        file_exists = os.path.isfile(CSV_FILE_PATH)
        with open(CSV_FILE_PATH, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                header = HEADER[:len(combined_data) +1]
                writer.writerow(header)
            # Append the title to the row and write it
            if len(RNN_buffer) == seq_len:
                for i in range(seq_len):
                    writer.writerow(RNN_buffer[i] + [CSV_TITLE])
                RNN_buffer = []
                print("Data written to CSV.")
            return True
    else:
        data = poll_data[0]
        nbr_gloves = 0
        if GLOVE_MODE == "DOUBLE":
            nbr_gloves = 2
        elif GLOVE_MODE == "SINGLE":
            nbr_gloves = 1
        if len(data) != nbr_gloves * feature_length:
            print(f"Skipping bad data: expected {nbr_gloves * feature_length}, got {len(data)} → {data}")
            return False
        # print_sensor_data_rnn(data)
        RNN_buffer.append(poll_data[0])
        if(len(RNN_buffer) == seq_len):
            translate_data()
            # print("Output: ", output)
            RNN_buffer = []
        return True
        
def parse_line_to_array(line):
    """
    Converts a comma-separated string of numbers into a list of floats.
    
    Args:
        line: A string received from the serial port.
        
    Returns:
        A list of floats if conversion is successful; otherwise, an empty list.
    """
    try:
        return [float(x) for x in line.split(',') if x.strip() != '']
    except ValueError:
        print("Error parsing line:", line)
        return []
    

def print_sensor_data_rnn(data: list):
    print(f"Flex sensors: {data[0]}, {data[1]}, {data[2]}, {data[3]}")
    print(f"Hall sensors: {data[4]}, {data[5]}, {data[6]}, {data[7]}")
    print(f"Accelerometer: x={data[8]}, y={data[9]}, z={data[10]}")
    print(f"Gyroscope: x={data[11]}, y={data[12]}, z={data[13]}")
    return


def read_serial_data():
    global WORD
    poll_data = []
    expected_arrays = 2 if GLOVE_MODE == "DOUBLE" else 1
    serial.Serial(COM_PORT, BAUD_RATE, timeout=1).close()
    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {COM_PORT} in {COMM_MODE} mode.")
            buffer = ''
            if TRAINING_MODE:
                for i in range(iterations):
                    input(f"Press ENTER when ready to sign {CSV_TITLE} Current iteration: {i +1}")
                    ser.reset_input_buffer()  # Flush old data
                    poll_data = []
                    samples_collected = 0

                    while samples_collected < seq_len:
                        data = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
                        if data:
                            buffer += data
                            while '\n' in buffer:
                                line_end = buffer.find('\n')
                                line = buffer[:line_end].strip()
                                buffer = buffer[line_end + 1:]

                                data_array = parse_line_to_array(line)
                                if data_array: #maybe do error checking here
                                    poll_data.append(data_array)
                                    if len(poll_data) == expected_arrays:
                                        if process_poll(poll_data):
                                            samples_collected += 1
                                        poll_data.clear()
                    else:
                        time.sleep(0.01)
            else: #testing
                while (1):
                    expected_arrays = 2 if GLOVE_MODE == "DOUBLE" else 1 
                     
                    # user_in = input(f"Press ENTER when ready ")
                    # if user_in == "p":
                    #     print(f"\033[35m{WORD}\033[0m")
                    #     continue
                    # if user_in == "d":
                    #     WORD = WORD[:-1]
                    #     print(f"\033[35m{WORD}\033[0m")
                    #     continue
                    ser.reset_input_buffer()  # Flush old data
                    poll_data = []
                    samples_collected = 0

                    while samples_collected < seq_len:
                        data = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
                        if data:
                            buffer += data
                            while '\n' in buffer:
                                line_end = buffer.find('\n')
                                line = buffer[:line_end].strip()
                                buffer = buffer[line_end + 1:]

                                data_array = parse_line_to_array(line)
                                if data_array:
                                    poll_data.append(data_array)
                                    if len(poll_data) == expected_arrays:
                                        if process_poll(poll_data):
                                            samples_collected += 1
                                        poll_data = []
                    else:
                        time.sleep(0.01)
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        print("Port closed")

# Prompt the user for the CSV title; this will be used as the file name (with .csv extension)
# and appended to each row.
def get_user_input():
    global CSV_FILE_PATH
    global CSV_SUBTITLE
    global CSV_TITLE
    
    CSV_TITLE = input("Enter CSV title (Sign): ")
    
    # if CSV_SUBTITLE == '':
    #     CSV_SUBTITLE = trainer_names[TRAINER_NAME]
    CSV_SUBTITLE = trainer_names[TRAINER_NAME]
    
    CSV_FILE_PATH = f"training_data/{CSV_TITLE}_{CSV_SUBTITLE}_{HAND}_dy.csv"
    print(CSV_SUBTITLE)
    return


    

def main():
    if TRAINING_MODE:
        global CSV_SUBTITLE
        global iterations
        # CSV_SUBTITLE = input("Enter CSV subtitle (trainer): ")
        iterations = int(input("Enter number of data points to be signed: "))
        while True: 
            get_user_input()
            read_serial_data()
    # In output mode, load the ML model.
    if not TRAINING_MODE:
        load_model()
    read_serial_data()

if __name__ == "__main__":
    main()