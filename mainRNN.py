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
import config
import threading
warnings.filterwarnings('ignore')

# -------------------------------
# Communication Port Configuration
# -------------------------------

BAUD_RATE = 115200           # Must match the ESP32's baud rate

# -------------------------------
# Global Variables for Data Sharing
# -------------------------------
LEFT_DATA = None
RIGHT_DATA = None

# Locks for thread-safe access
left_lock = threading.Lock()
right_lock = threading.Lock()


# -------------------------------
# File, Model, and Sensor Settings
# -------------------------------
CSV_FILE_PATH = None  # Will be set based on user input if TRAINING_MODE is True
CSV_TITLE = None      # Global title for the CSV
CSV_SUBTITLE = None
iterations = 100
model = None
scaler = None
label_encoder  = None
seq_len = 20
WORD = ''
feature_length = 15
prediction_buffer = []
BUFFER_SIZE = 3
threshold = -8.8

MODEL_PATH_LOCAL = None
LABEL_ENCODER_PATH = None
SCALER_PATH = None


flex = 3200
# threshold = -7.5

# Predefined sensor names (modify as needed for your setup)
HEADER = ["hall_1", "hall_2", "hall_3", "flex_t", "flex_i", "flex_m", "flex_r", "flex_p", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "hand", "sign"]
HEADER_DOUBLE = ["hall_1_L", "hall_2_L", "hall_3_L", "flex_t_L", "flex_i_L", "flex_m_L", "flex_r_L", "flex_p_L", "accel_x_L", "accel_y_L", "accel_z_L", "gyro_x_L", "gyro_y_L", "gyro_z_L", "hand", "hall_1_R", "hall_2_R", "hall_3_R", "flex_t_R", "flex_i_R", "flex_m_R", "flex_r_R", "flex_p_R", "accel_x_R", "accel_y_R", "accel_z_R", "gyro_x_R", "gyro_y_R", "gyro_z_R", "hand", "sign"]

def load_model(hand):
    """
    Loads the PyTorch model from the specified MODEL_PATH_LOCAL.
    """
    global model
    global scaler
    global label_encoder
    
    if hand == "clear":
        print(f"Clearing loaded model and Buffer")
        prediction_buffer.clear()
        model = None
        scaler = None
        label_encoder = None
        return
    try:
        
        model = torch.load(os.path.join(config.COMMON_PATH, f"RNN_model_{hand}.pth"), weights_only=False)
        model.eval()
        label_encoder = joblib.load(os.path.join(config.COMMON_PATH, f"label_encoder_{hand}.pkl"))
        scaler = joblib.load(os.path.join(config.COMMON_PATH, f"scaler_{hand}.pkl"))
  
        
        print(f"{os.path.join(config.COMMON_PATH, f"RNN_model_{hand}.pth")} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")


def update_prediction_buffer(predicted_label):
    global WORD, prediction_buffer

    sign = str(predicted_label[0]) 

    prediction_buffer.append(sign)

    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)
  
    print(f"Prediction Buffer: {prediction_buffer}\n")
    
    if len(prediction_buffer) == BUFFER_SIZE and len(set(prediction_buffer)) == 1:
        # print(f"\033[33mWait for Model load\033[0m")
        WORD += sign
        prediction_buffer.clear()  
   
    
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

        normalized_buffer = scaler.transform(RNN_buffer)
        tensor_input = torch.tensor(normalized_buffer, dtype=torch.float32).unsqueeze(0)

        
        
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
            print(f"\n\033[32mOutput: {predicted_label[0]} with confidence {max_prob.item() * 100:.2f}%\033[0m")
            update_prediction_buffer(predicted_label)


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
    global RNN_buffer
    if config.GLOVE_MODE == "DOUBLE" and not config.TRAINING_MODE:
        if not MODEL_PATH_LOCAL:
            return True
        elif "_D" in MODEL_PATH_LOCAL:
            combined_data = poll_data[0] + poll_data[1]
        elif "_L" in MODEL_PATH_LOCAL:
            combined_data = poll_data[0] 
        elif "_R" in MODEL_PATH_LOCAL:
            combined_data = poll_data[1] 
    elif config.GLOVE_MODE == "DOUBLE":
        combined_data = poll_data[0] + poll_data[1]     
        print(combined_data) 
    else:  # SINGLE mode
        combined_data = poll_data[0]

    if config.TRAINING_MODE:
        RNN_buffer.append(combined_data)
        # Write to CSV with the CSV_TITLE appended.
        file_exists = os.path.isfile(CSV_FILE_PATH)
        with open(CSV_FILE_PATH, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                if(config.GLOVE_MODE == "DOUBLE"):
                    writer.writerow(HEADER_DOUBLE)
                else:
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
    
        if combined_data:
            RNN_buffer.append(combined_data)
            if(len(RNN_buffer) == seq_len):
                translate_data()
                RNN_buffer = []
            return True
        else:
            print("NO COMBINED DATA??")
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


def update_glove_mode():
    global LEFT_DATA, RIGHT_DATA, MODEL_PATH_LOCAL, LABEL_ENCODER_PATH, SCALER_PATH, threshold

    global RNN_buffer
    RNN_buffer = []

    with left_lock:
        left = LEFT_DATA
    with right_lock:
        right = RIGHT_DATA

    if left and right:
        if left[14] == 1:
           print(f"\033[31mERROR: {config.LEFT_COM} for Left Glove is connected to the Right Glove\033[0m")
        print(f"[LEFT] y accelerometer: {left[9]}")
        print(f"[LEFT] Flex sensors: {left[3]}, {left[4]}, {left[5]}, {left[6]}, {left[7]}")

        print(f"[RIGHT] y accelerometer: {right[9]}")
        print(f"[RIGHT] Flex sensors: {right[3]}, {right[4]}, {right[5]}, {right[6]}, {right[7]}")
        y_left = left[9]
        y_right = right[9]

        
        flex = 3200
        if (is_glove_deactivated(left) or is_glove_deactivated(right)) :
            
            # Use only the active glove
            if y_left >= threshold:
                if MODEL_PATH_LOCAL and MODEL_PATH_LOCAL.endswith("RNN_model_L.pth"):
                    print("\033[34mStaying on Left Model\033[0m")
                else:    
                    print("\033[35mSwitching to Left Model\033[0m")
                    MODEL_PATH_LOCAL = os.path.join(config.COMMON_PATH, f"RNN_model_L.pth")
                    LABEL_ENCODER_PATH = os.path.join(config.COMMON_PATH, f"label_encoder_L.pkl")
                    SCALER_PATH = os.path.join(config.COMMON_PATH, f"scaler_L.pkl")
                    prediction_buffer.clear()
                    load_model("L")
                    print(f"\033[33mWait for Model load\033[0m")
                    time.sleep(2)

            elif y_right >= threshold:
                if MODEL_PATH_LOCAL and MODEL_PATH_LOCAL.endswith("RNN_model_R.pth"):
                    print("\033[34mStaying on Right Model\033[0m")
                else:  
                    print("\033[35mSwitching to Right Model\033[0m")
                    MODEL_PATH_LOCAL = os.path.join(config.COMMON_PATH, f"RNN_model_R.pth")
                    LABEL_ENCODER_PATH = os.path.join(config.COMMON_PATH, f"label_encoder_R.pkl")
                    SCALER_PATH = os.path.join(config.COMMON_PATH, f"scaler_R.pkl")
                    prediction_buffer.clear()
                    load_model("R")
                    print(f"\033[33mWait for Model load\033[0m")
                    time.sleep(2)
                
            else:
                print("\033[35mBoth gloves deactivated\033[0m")
                MODEL_PATH_LOCAL = None
                LABEL_ENCODER_PATH = None
                SCALER_PATH = None
                load_model("clear")

        else:
            if MODEL_PATH_LOCAL and MODEL_PATH_LOCAL.endswith("RNN_model_D.pth"):
                print("\033[34mStaying on Double Model\033[0m")
            else:
                print("\033[35mSwitching to Double Model\033[0m")
                MODEL_PATH_LOCAL = os.path.join(config.COMMON_PATH, f"RNN_model_D.pth")
                LABEL_ENCODER_PATH = os.path.join(config.COMMON_PATH, f"label_encoder_D.pkl")
                SCALER_PATH = os.path.join(config.COMMON_PATH, f"scaler_D.pkl")
                prediction_buffer.clear()
                load_model("D")
                print(f"\033[33mWait for Model load\033[0m")
                time.sleep(2)

    else:
        print("\033[31mINVALID DATA\033[0m")

        MODEL_PATH_LOCAL = None
        LABEL_ENCODER_PATH = None
        SCALER_PATH = None
        load_model("clear")


def Collect_double_glove_data():
    global  LEFT_DATA, RIGHT_DATA
   
    if config.TRAINING_MODE:
        for i in range(iterations):
            print("Assume the position")
            time.sleep(2)
            print("Begin Signing")
            # input(f"Press ENTER when ready to sign \"{CSV_TITLE}\"  Current iteration: {i + 1}")
            poll_data = []
            samples_collected = 0

            with left_lock:
                LEFT_DATA = None
            with right_lock:
                RIGHT_DATA = None

            while samples_collected < seq_len:
                
                with left_lock:
                    left = LEFT_DATA
                with right_lock:
                    right = RIGHT_DATA

                if left and right:
                    poll_data = [left, right]
                  
                    if process_poll(poll_data): 
                        samples_collected += 1
                    with left_lock:
                        LEFT_DATA = None
                    with right_lock:
                        RIGHT_DATA = None
    
            # time.sleep(0.01) 
    else:
        poll_data = []
        samples_collected = 0
       
        while samples_collected < seq_len:
    
            with left_lock:
                left = LEFT_DATA
            with right_lock:
                right = RIGHT_DATA

            if left and right:
                poll_data = [left, right]
                if process_poll(poll_data): 
                    samples_collected += 1
                
                if samples_collected != 20:
                    with left_lock:
                        LEFT_DATA = None
                    with right_lock:
                        RIGHT_DATA = None

def read_serial_data_double(com_port, hand):
    global LEFT_DATA, RIGHT_DATA
  
    try:
        ser = serial.Serial(com_port, BAUD_RATE, timeout=1)
        print(f"[{hand}] Connected to {com_port}.")
        buffer = ''
        while True:
            # ser.reset_input_buffer()
            data = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
       
            if data:
                buffer += data
                while '\n' in buffer:
                    line_end = buffer.find('\n')
                    line = buffer[:line_end].strip()
                    buffer = buffer[line_end + 1:]
                    data_array = parse_line_to_array(line)
              
                    if len(data_array) == feature_length:
                        if hand == "LEFT":
                            with left_lock:
                                LEFT_DATA = data_array
                        else:
                            with right_lock:
                                RIGHT_DATA = data_array
                    else:
                        print(f"[{hand}] Invalid data length: {line} in COMPORT: {com_port}")
    except serial.SerialException as e:
        print(f"Error opening {com_port}: {e}")

def read_serial_data_single():
    global WORD, RNN_buffer
    poll_data = []
    expected_arrays = 2 if config.GLOVE_MODE == "DOUBLE" else 1
    printed = False  # For testing mode, handles glove deactivation printing

    try:
        with serial.Serial(config.LEFT_COM, BAUD_RATE, timeout=1) as ser:
            print(f"Connected to {config.LEFT_COM}")
            buffer = ''

            if config.TRAINING_MODE:
                for i in range(iterations):
                    input(f"Press ENTER when ready to sign {CSV_TITLE} Current iteration: {i + 1}")
                    ser.reset_input_buffer()
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
                                if len(data_array) == feature_length:
                                    poll_data.append(data_array)
                                    if len(poll_data) == expected_arrays:
                                        if process_poll(poll_data):
                                            samples_collected += 1
                                        poll_data.clear()
                                else:
                                    print(f"Invalid data length: {line}")
                    time.sleep(0.01)

            else:  # TESTING MODE
                expected_arrays = 1  # SINGLE mode
                while True:
                    ser.reset_input_buffer()
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
                                if len(data_array) == feature_length:
                                    poll_data.append(data_array)
                                    if len(poll_data) == expected_arrays:
                                        if process_poll(poll_data):
                                         
                                            if printed:
                                                RNN_buffer = []
                                            if not printed:
                                                if (is_glove_deactivated(data_array)):
                                                    RNN_buffer = []
                                                    print("\033[35mGloves deactivated\033[0m")
                                                    print(f"y accelerometer: {data_array[9]}")
                                                    print(f"Flex sensors: {data_array[3]}, {data_array[4]}, {data_array[5]}, {data_array[6]}, {data_array[7]}")
                                                    printSign()
                                                    printed = True

                                            elif not is_glove_deactivated(data_array):
                                                # print("is active")
                                                printed = False

                                            samples_collected += 1
                                        poll_data = []
                                else:
                                    print(f"Invalid data length: {line}")

                    time.sleep(0.01)

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        print("Port closed")

def is_glove_deactivated(data_array):
    return (
        (data_array[9] < threshold and data_array[3] > flex and data_array[4] > flex and 
         data_array[5] > 2500 and data_array[6] > flex and data_array[7] > flex and data_array[14] == 0)
        or
        (data_array[9] < threshold and data_array[3] > 3000 and data_array[4] > flex and 
         data_array[5] > flex and data_array[6] > flex and data_array[7] > flex and data_array[14] == 1)
    )


# Prompt the user for the CSV title; this will be used as the file name (with .csv extension)
# and appended to each row.
def get_user_input():
    global CSV_FILE_PATH
    global CSV_SUBTITLE
    global CSV_TITLE
    
    CSV_TITLE = input("Enter CSV title (Sign): ")
    
    # if CSV_SUBTITLE == '':
    #     CSV_SUBTITLE = trainer_names[TRAINER_NAME]
    CSV_SUBTITLE = config.trainer_names[config.TRAINER_NAME]
    
    CSV_FILE_PATH = f"training_data/{CSV_TITLE}_{CSV_SUBTITLE}_{config.HAND}_dy.csv"
    print(CSV_SUBTITLE)
    print(CSV_FILE_PATH)
    return

def printSign():
    global MODEL_PATH_LOCAL
    if not MODEL_PATH_LOCAL: #only print if hands are down
        print(f"\033[38;2;255;165;0mFull Sign: {WORD}\033[0m")


    

def mainsingle():
    if config.TRAINING_MODE:
        global CSV_SUBTITLE
        global iterations
        # CSV_SUBTITLE = input("Enter CSV subtitle (trainer): ")
        iterations = int(input("Enter number of data points to be signed: "))
        while True: 
            get_user_input()
            read_serial_data_single()
    # In output mode, load the ML model.
    if not config.TRAINING_MODE:
        load_model(config.HAND)
    read_serial_data_single()




def maindouble():
    if config.TRAINING_MODE:
        global CSV_SUBTITLE, iterations
        
        iterations = int(input("Enter number of data points to be signed: "))
        
        print(CSV_SUBTITLE)

        left_thread = threading.Thread(target=read_serial_data_double, args=(config.LEFT_COM, "LEFT"), daemon= True)
        right_thread = threading.Thread(target=read_serial_data_double, args=(config.RIGHT_COM, "RIGHT"), daemon= True)
    
        left_thread.start()
        right_thread.start()

        # Run training loop
        while True: 
            get_user_input()
            Collect_double_glove_data()
    
    else:
     

        left_thread = threading.Thread(target=read_serial_data_double, args=(config.LEFT_COM, "LEFT"), daemon= True)
        right_thread = threading.Thread(target=read_serial_data_double, args=(config.RIGHT_COM, "RIGHT"), daemon= True)
       
        left_thread.start()
        right_thread.start()

        time.sleep(1)

     
        while True:
            update_glove_mode()
            printSign()
            if MODEL_PATH_LOCAL != None:
                print(f"\033[33mStart Signing\033[0m")

            Collect_double_glove_data()
           


if __name__ == "__main__":
    if config.GLOVE_MODE == "DOUBLE":
        maindouble()
    else:
        mainsingle()