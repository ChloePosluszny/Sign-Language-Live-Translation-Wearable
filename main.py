import serial
import time
import csv
import torch  # PyTorch required for loading .pth models
import joblib #Required to load MLP Classifier from sklearn
import numpy as np
import os

# -------------------------------
# Global Mode Flags (toggle as needed)
# -------------------------------
TRAINING_MODE = True          # True: training (write to CSV), False: output (ML inference)
COMM_MODE = "SERIAL"       # Options: "BLUETOOTH" or "SERIAL"
GLOVE_MODE = "SINGLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)

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
MODEL_PATH = "mlp_translation_model.pkl"
model = None

# Predefined sensor names (modify as needed for your setup)
SENSOR_NAMES_SINGLE = ["hall_1", "hall_2", "hall_3", "flex_t", "flex_i", "flex_m", "flex_r", "flex_p", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
SENSOR_NAMES_DOUBLE = SENSOR_NAMES_SINGLE + [name + "_R" for name in SENSOR_NAMES_SINGLE]

def load_model():
    """
    Loads the PyTorch model from the specified MODEL_PATH.
    """
    global model
    try:
        # model = torch.load(MODEL_PATH)
        # model.eval()
        model = joblib.load(MODEL_PATH)
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
        converted_data = np.array(poll_data)
        converted_data = converted_data.reshape(1,-1)
        prediction = model.predict(converted_data)
        return prediction[0]

def process_poll(poll_data):
    """
    Processes a complete poll of sensor data based on the current mode.
    
    For training, it combines the arrays (if DOUBLE mode, concatenates the two arrays)
    and appends them to a CSV file with the CSV title appended to both the header and each row.
    For output mode, it sends the data to the ML model.
    
    Args:
        poll_data: List of arrays from the glove(s).
    """
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
        output = translate_data(poll_data)
        print("Output:", output)

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

    try:
        with serial.Serial(COM_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {COM_PORT} in {COMM_MODE} mode.")
            buffer = ''
            while True:
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

if __name__ == "__main__":
    if TRAINING_MODE:
        # Prompt the user for the CSV title; this will be used as the file name (with .csv extension)
        # and appended to each row.
        CSV_TITLE = input("Enter CSV title: ")
        CSV_FILE_PATH = f"training_data/{CSV_TITLE}.csv"
    # In output mode, load the ML model.
    if not TRAINING_MODE:
        load_model()
    read_serial_data()