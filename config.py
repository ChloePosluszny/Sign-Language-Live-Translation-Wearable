import os
trainer_names = {'ALFONSO': 'a', 'CHLOE': 'c', 'DAVID': 'd', 'ERIK': 'e', 'RAHMAN': 'r', 'TEST': 't'}

TRAINING_MODE = True       # True: training (write to CSV), False: output (ML inference)
TRAINER_NAME = 'TEST'
LEFT_COM = 'COM3'
RIGHT_COM = 'COM6'
GLOVE_MODE = "DOUBLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)
HAND = "D" #FOR SINGLE MODE

COMMON_PATH = os.path.join("Models", trainer_names[TRAINER_NAME])

MODEL_PATH = f"Models/{trainer_names[TRAINER_NAME]}/RNN_model_{HAND}.pth" #FOR SINGLE MODE