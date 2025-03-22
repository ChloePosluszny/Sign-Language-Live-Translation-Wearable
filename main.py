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
from RNN_translator import RNN
import warnings
warnings.filterwarnings('ignore')


trainer_names = {'ALFONSO': 'a', 'CHLOE': 'c3', 'DAVID': 'd', 'ERIK': 'e', 'RAHMAN': 'r'}

# -------------------------------
# Global Mode Flags (toggle as needed)
# -------------------------------
TRAINING_MODE = False        # True: training (write to CSV), False: output (ML inference)
TRAINER_NAME = 'CHLOE'
COMM_MODE = "SERIAL"       # Options: "BLUETOOTH" or "SERIAL"
GLOVE_MODE = "SINGLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)

# -------------------------------
# Communication Port Configuration
# -------------------------------
BLUETOOTH_COM_PORT = 'COM5'   # Port for Bluetooth
SERIAL_COM_PORT = '/dev/cu.usbserial-0001'      # Port for direct serial connection (e.g., USB)

# Select the appropriate COM port based on COMM_MODE
COM_PORT = BLUETOOTH_COM_PORT if COMM_MODE == "BLUETOOTH" else SERIAL_COM_PORT
BAUD_RATE = 115200           # Must match the ESP32's baud rate

# -------------------------------
# File, Model, and Sensor Settings
# -------------------------------
CSV_FILE_PATH = None  # Will be set based on user input if TRAINING_MODE is True
CSV_TITLE = None      # Global title for the CSV
TRAINING_TIME = 12
MODEL_PATH = "RNN_model.pth"
model = None
scaler = None
label_encoder  = None

# Predefined sensor names (modify as needed for your setup)
SENSOR_NAMES_SINGLE = ["hall_1", "hall_2", "hall_3", "flex_t", "flex_i", "flex_m", "flex_r", "flex_p", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
SENSOR_NAMES_DOUBLE = SENSOR_NAMES_SINGLE + [name + "_R" for name in SENSOR_NAMES_SINGLE]

def load_model():
    """
    Loads the PyTorch model from the specified MODEL_PATH.
    """
    global model
    global scaler
    global label_encoder
    print(f"Attempting to load model from: {MODEL_PATH}")
    try:
        model = torch.load(MODEL_PATH, weights_only=False)
        model.eval()
        label_encoder = joblib.load("label_encoder.pkl")
  

        # model = joblib.load(MODEL_PATH)
        # scaler = joblib.load("scaler.pkl")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def translate_data(poll_data):
    """
    Placeholder function for translating glove sensor data using a machine learning model.
    
    Args:
        poll_data: A list containing one or two lists of floats (data from glove(s)).
        
    Returns:
        A string representing the translated output.
    """
    if model is None:
        return "Model not loaded"
    # Example: Convert poll_data to a tensor, process it with the model, then decode the result.
    else:
        # if len(poll_data[0]) != 14:
        #     return "not 14"
        
        # poll_data = scaler.transform(poll_data)
        
        #test to see if works
        # df_data = pd.DataFrame(poll_data, columns=SENSOR_NAMES_SINGLE)
        # prediction = model.predict(df_data)
        # return prediction[0]

        # converted_data = np.array(poll_data)
        # # probabilities = model.predict_proba(converted_data)
        # converted_data = converted_data.reshape(1,-1)
        # prediction = model.predict(converted_data)
        # return prediction[0]
    
        #poll data should be a list of lists of sequence length data entries
        tensor_input = torch.tensor(poll_data, dtype=torch.float32).unsqueeze(0)
        if tensor_input.size(-1) != 14:
            print(f"Expected 14 features, but got {tensor_input.size(-1)}")
            return "Invalid input shape"

        with torch.no_grad():
            outputs = model(tensor_input)
            _, predicted = torch.max(outputs, 1)
            predicted_label = label_encoder.inverse_transform(predicted)
            return predicted_label[0]
        
           
            



translations = []
# old_letter = "*"
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
    if TRAINING_MODE:
        if GLOVE_MODE == "DOUBLE":
            combined_data = poll_data[0] + poll_data[1]
        else:  # SINGLE mode
            combined_data = poll_data[0]
        
        # Write to CSV with the CSV_TITLE appended.
        file_exists = os.path.isfile(CSV_FILE_PATH)
        with open(CSV_FILE_PATH, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                if GLOVE_MODE == "DOUBLE":
                    header = SENSOR_NAMES_DOUBLE[:len(combined_data)]
                else:
                    header = SENSOR_NAMES_SINGLE[:len(combined_data)]
                header.append('sign')  # Append the title at the end of the header
                writer.writerow(header)
            # Append the title to the row and write it
            writer.writerow(combined_data + [CSV_TITLE])
        print("Data written to CSV.")
    else:
        # Output mode: process the data using the ML model.
        #test for 1 data entries rn
        # if len(poll_data[0]) == 14:
        #     output = translate_data(poll_data)
        #     print("Output: ", output)
        # else:
        #     print(f"poll data has {len(poll_data)} features")
        # if output != old_letter:
        #     print("Output:", output)
        #     old_letter = output
            # print("probs", proba)
       
        output = translate_data(poll_data)
        print("Output: ", output)
        # translations.append(output)
        # if len(translations) == 10:
        #     print("Output:", max(set(translations), key=translations.count))
        #     translations.clear()
        
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

def read_serial_data():
    """
    Connects to the specified COM port (Bluetooth or Serial) and continuously reads data.
    
    Depending on GLOVE_MODE, it expects either one or two 1D arrays per poll. Once the
    expected number of arrays is received, the poll is processed according to TRAINING_MODE.
    """
    poll_data = []
    expected_arrays = 2 if GLOVE_MODE == "DOUBLE" else 1
    
    serial.Serial(COM_PORT, BAUD_RATE, timeout=1).close()

    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {COM_PORT} in {COMM_MODE} mode.")
            buffer = ''
            if TRAINING_MODE:
                start_time = time.time()
            while TRAINING_MODE == 0 or time.time() - start_time < TRAINING_TIME:
                data = ser.read_all().decode('utf-8', errors='ignore')
                if data:
                    buffer += data
                    # Process complete lines from the buffer.
                    while '\n' in buffer:
                        line_end = buffer.find('\n')
                        line = buffer[:line_end].strip()
                        buffer = buffer[line_end + 1:]
                        if line:
                            data_array = parse_line_to_array(line)
                            if data_array:
                                poll_data.append(data_array)
                                # Once we've collected the expected arrays, process the poll.
                                if len(poll_data) == expected_arrays:
                                    process_poll(poll_data)
                                    poll_data = []  # Reset for the next poll
                else:
                    time.sleep(0.01)  # Prevent busy waiting
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        print("Port closed")

# Prompt the user for the CSV title; this will be used as the file name (with .csv extension)
# and appended to each row.
def get_user_input():
    global CSV_FILE_PATH
    global TRAINING_TIME
    global CSV_TITLE
    global CSV_SUBTITLE
    CSV_TITLE = input("Enter CSV title: ")
    CSV_SUBTITLE = input("Enter CSV subtitle: ")
    timer = input("Enter training time (in seconds):")
    if CSV_SUBTITLE == '':
        CSV_SUBTITLE = trainer_names[TRAINER_NAME]
    if timer != '':
        TRAINING_TIME = int(timer)
    CSV_FILE_PATH = f"training_data/{CSV_TITLE}_{CSV_SUBTITLE}.csv"
    return

if __name__ == "__main__":
    if TRAINING_MODE:
        while True: 
            get_user_input()
            read_serial_data()
    # In output mode, load the ML model.
    if not TRAINING_MODE:
        load_model()
    read_serial_data()