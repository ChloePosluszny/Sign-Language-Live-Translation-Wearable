import os
trainer_names = {'ALFONSO': 'a', 'CHLOE': 'c', 'DAVID': 'd', 'ERIK': 'e', 'RAHMAN': 'r', 'TEST': 't'}

TRAINING_MODE = False       # True: training (write to CSV), False: output (ML inference)
#when training in double glove mode after you enter iteration it may give a punch of prints from thread of reading com ports just type sign and press enter to continue
TRAINER_NAME = 'DAVID'
COMM_MODE = "SERIAL"       # Options: "BLUETOOTH" or "SERIAL"
GLOVE_MODE = "SINGLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)
HAND = "R"
TIMER_MODE = True
LEFT_COM = 'COM3'
RIGHT_COM = 'COM7'
GLOVE_MODE = "DOUBLE"         # Options: "DOUBLE" (expect 2 arrays) or "SINGLE" (expect 1 array)
HAND = "D" # captial D, R, or L depending on hand being used


COMMON_PATH = os.path.join("Models", trainer_names[TRAINER_NAME])

MODEL_PATH = f"Models/{trainer_names[TRAINER_NAME]}/RNN_model_{HAND}.pth" #FOR SINGLE MODE