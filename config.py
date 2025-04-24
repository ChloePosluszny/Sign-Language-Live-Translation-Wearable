# config.py
import os

trainer_names = {'ALFONSO': 'a', 'CHLOE': 'c', 'DAVID': 'd', 'ERIK': 'e', 'RAHMAN': 'r', 'TEST': 't'}

TRAINING_MODE = False
TRAINER_NAME  = 'DAVID'
COMM_MODE     = "SERIAL"
GLOVE_MODE    = "SINGLE"
HAND          = "L"

COMMON_PATH = os.path.join("Models", trainer_names[TRAINER_NAME])
os.makedirs(COMMON_PATH, exist_ok=True)

MODEL_PATH = os.path.join(COMMON_PATH, f"RNN_model_{HAND}.pth")
