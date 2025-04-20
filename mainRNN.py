import serial
import time
import csv
import torch  # PyTorch required for loading .pth models
import joblib  # Required to load MLP Classifier from sklearn
import numpy as np
import os
import threading
import queue
import glob
from Train_RNN import RNN
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Global Configuration
# -------------------------------
trainer_names = {'ALFONSO': 'a1', 'CHLOE': 'c3', 'DAVID': 'd1', 'ERIK': 'e', 'RAHMAN': 'r'}

# Mode flags
TRAINING_MODE = False        # True: training (write to CSV), False: output (ML inference)
TRAINER_NAME = 'RAHMAN'

COMM_MODE = "SERIAL"        # Options: "BLUETOOTH" or "SERIAL"
GLOVE_MODE = "DOUBLE"       # Options: "DOUBLE" (2 gloves) or "SINGLE"
HAND = "R"

# Serial ports
LEFT_COM = 'COM3'   # Left glove on serial USB
RIGHT_COM = 'COM5'  # Right glove on Bluetooth port

BAUD_RATE = 115200

# Training / Model settings
CSV_FILE_PATH = None
CSV_TITLE = None
CSV_SUBTITLE = None
iterations = 100
MODEL_PATH_LEFT = "RNN_model_L.pth"
MODEL_PATH_RIGHT = "RNN_model_R.pth"
MODEL_PATH_DOUBLE = "RNN_model_D.pth"
mode_left = None
scaler_left = None
label_encoder = None
seq_len = 20
feature_length = 15

# Fixed header for CSV
HEADER = [
    "hall_1", "hall_2", "hall_3",
    "flex_t", "flex_i", "flex_m", "flex_r", "flex_p",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
    "hand", "sign"
]
ACCEL_Y_IDX = HEADER.index("accel_y")
ACCEL_Y_THRESHOLD = -9.0  # m/s^2 threshold to deactivate a glove

# Queues for incoming data    
data_q = queue.Queue()

# Buffer for RNN sequences
RNN_buffer = []

def load_model():
    """Loads the PyTorch model, scaler, and label encoder."""
    global model, scaler, label_encoder
    print(f"Attempting to load model from: {MODEL_PATH}")
    try:
        model = torch.load(MODEL_PATH, weights_only=False)
        model.eval()
        label_encoder = joblib.load(f"label_encoder_{HAND}.pkl")
        scaler = joblib.load(f"scaler{HAND}.pkl")
        print(f"{MODEL_PATH} loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def translate_data():
    """Runs inference on RNN_buffer when full."""
    global RNN_buffer
    if model is None:
        print("Model not loaded")
        return
    # Normalize and reshape
    normalized = scaler.transform(RNN_buffer)
    tensor_input = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
    if tensor_input.size(-1) != feature_length:
        print("Invalid input shape")
        return
    with torch.no_grad():
        outputs = model(tensor_input)
        probs = torch.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, dim=1)
        classes = label_encoder.classes_
        for lbl, p in zip(classes, probs[0]):
            print(f"Label: {lbl}, Probability: {p.item()*100:.2f}%")
        pred_label = label_encoder.inverse_transform(predicted)[0]
        print(f"\n\033[32mOutput: {pred_label} ({max_prob.item()*100:.2f}%)\033[0m")
    RNN_buffer = []

def process_poll(poll_data):
    """Handles one poll: training (writes CSV) or inference."""
    global RNN_buffer
    # Combine data arrays based on count
    if len(poll_data) == 2:
        combined = poll_data[0] + poll_data[1]
    elif len(poll_data) == 1:
        combined = poll_data[0]
    else:
        return False

    # Ensure correct feature count
    if TRAINING_MODE:
        if len(combined) != feature_length:
            return False
        RNN_buffer.append(combined)
        # Write out when buffer full
        if len(RNN_buffer) == seq_len:
            # Write CSV
            os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
            file_exists = os.path.isfile(CSV_FILE_PATH)
            with open(CSV_FILE_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(HEADER[:feature_length] + ["sign"])
                for row in RNN_buffer:
                    writer.writerow(row + [CSV_TITLE])
            print("Data written to CSV.")
            RNN_buffer = []
        return True
    else:
        # Inference
        if len(poll_data) != (2 if GLOVE_MODE=="DOUBLE" else 1):
            return False
        RNN_buffer.append(combined)
        if len(RNN_buffer) == seq_len:
            translate_data()
        return True

def reader_thread(port, hand_label):
    """Read from a COM port, parse lines, and push to queue."""
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    buf = ""
    while True:
        chunk = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
        if chunk:
            buf += chunk
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                parts = [p for p in line.strip().split(',') if p]
                if len(parts) == feature_length:
                    row = list(map(float, parts))
                    data_q.put((hand_label, row))

def read_serial_data():
    """Dispatch single or double glove reading."""
    if not TRAINING_MODE:
        load_model()

    if GLOVE_MODE == "DOUBLE":
        # Start both readers
        threading.Thread(target=reader_thread, args=(LEFT_COM, "L"), daemon=True).start()
        threading.Thread(target=reader_thread, args=(RIGHT_COM, "R"), daemon=True).start()
        last = {"L": None, "R": None}

        while True:
            hand, row = data_q.get()  # blocking
            last[hand] = row

            # Determine active gloves
            active = []
            for h in ["L", "R"]:
                if last[h] is None:
                    continue
                # deactivate if below threshold
                if last[h][ACCEL_Y_IDX] < ACCEL_Y_THRESHOLD:
                    continue
                active.append(last[h])

            if not active:
                continue

            process_poll(active)

    else:
        # Single glove: use existing COM_PORT
        COM_PORT = LEFT_COM
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        buf = ""
        while True:
            chunk = ser.read(ser.in_waiting or 1).decode('utf-8', errors='ignore')
            if not chunk:
                time.sleep(0.01)
                continue
            buf += chunk
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                parts = [p for p in line.strip().split(',') if p]
                if len(parts) != feature_length:
                    continue
                row = list(map(float, parts))
                process_poll([row])

def get_user_input():
    """Prompt for CSV title and set path."""
    global CSV_TITLE, CSV_SUBTITLE, CSV_FILE_PATH, iterations
    CSV_TITLE = input("Enter CSV title (Sign): ").strip()
    CSV_SUBTITLE = trainer_names[TRAINER_NAME]
    CSV_FILE_PATH = f"training_data/{CSV_TITLE}_{CSV_SUBTITLE}_{HAND}_dy.csv"
    iterations = int(input("Enter number of data points to be signed: "))
    print(f"Logging {iterations} samples of '{CSV_TITLE}' by {TRAINER_NAME}")

def main():
    if TRAINING_MODE:
        get_user_input()
    read_serial_data()

if __name__ == "__main__":
    main()