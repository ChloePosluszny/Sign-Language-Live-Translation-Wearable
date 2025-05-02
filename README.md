# Sign Language Translation Wearable
The Sign Language Translation Wearable (SLTW) is an open-source project centered around a pair of gloves that allows for live translation of the American Sign Language (ASL) alphabet and a curated list of ASL words and phrases into written ASL text. The gloves consists of several sensors that is read by an ESP32, which then sends the data to a computer over a communication port. This data is then fed into a machine learning algorithm using a Recurrent Neural Network (RNN) which translates the sign and outputs the text. This repo contains the software used in the SLTW project and is free to be used and modified by anyone who is interested. The rest of this README will explain how to setup and use this software, as well as document the various scripts and programs used to operate the SLTW.

## Setup
The first step in setting up the glove software is flashing the ESP32. First, open [integrated.ino](integrated/integrated.ino) in the Arduino IDE (or your preferred software). You will need to install the ESP32 board and the necessary libraries included in the header (TODO: expand on this). Once the program is able to be compiled, flash it to the ESP32.

The next step is to setup the python environment. Install the libraries included in the [requriements.txt](requirements.txt) using your preferred method. Modify [config.py](config.py) to reflect your setup. The SLTW programs are now ready to be run. Instructions to run these programs follows.

## Process
This section will explain the process in collecting data, training the RNN, and translating the data.

### Collecting Data and Training
Before the gloves can translate, it needs a database full of signs for the RNN to train on. While a database is already included, you will want to provide data of your own to better fit the way you sign (and it helps provide variation in training for everyone else). To collect data, go into [config.py](config.py) and set `TRAINING_MODE = TRUE`, and set the other parameters as appropriate (TODO). Run `python mainRNN.py` and follow the instructions to begin signing the letters. This will put data into the database. Once done collecting the desired signs, run `python concat.py` to concatenate all the data. You may want to verify the data was concatenated properly. Then, run `python Train_RNN.py` to train the machine learning model.

### Translating
Once the machine learning model has been trained, go into config.py and set `TRAINING_MODE = FALSE`. Run `python mainRNN.py` to begin running the translation.

## Troubleshooting
Some common problems that may occur
- The gloves may be in the wrong com ports. To fix this, make sure the left glove is plugged in first, and then plug in the right glove
- When starting the program, a large amount of "Bad header" is printed. To fix this, fully flex the gloves for a few seconds
