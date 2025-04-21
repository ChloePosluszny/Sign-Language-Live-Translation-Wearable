import os
trainer_names = {'ALFONSO': 'a', 'CHLOE': 'c', 'DAVID': 'd', 'ERIK': 'e', 'RAHMAN': 'r', 'TEST': 't'}

TRAINING_MODE = False       # True: training (write to CSV), False: output (ML inference)
TRAINER_NAME = 'DAVID'
COMM_MODE = "SERIAL"       # Options: "BLUETOOTH" or "SERIAL"
GLOVE_MODE = "SINGLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)
HAND = "R"

COMMON_PATH = os.path.join("Models", trainer_names[TRAINER_NAME])

MODEL_PATH = f"Models/{trainer_names[TRAINER_NAME]}/RNN_model_{HAND}.pth"