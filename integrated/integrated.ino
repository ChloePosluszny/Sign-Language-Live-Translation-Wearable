#define PACKET_DELAY 50  // Milliseconds between packets
#define MAX_PACKET_SIZE 64
#define NUM_HALL 3
#define NUM_FLEX 5

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include "BluetoothSerial.h"

const bool BLUETOOTH = false; // Set to true to enable Bluetooth output

BluetoothSerial SerialBT;
Adafruit_MPU6050 mpu;

void myPrint(const String &message) {
  if (BLUETOOTH) {
    SerialBT.print(message);
  } else {
    Serial.print(message);
  }
}

void myPrintln(const String &message = "") {
  if (BLUETOOTH) {
    SerialBT.println(message);
  } else {
    Serial.println(message);
  }
}

const int flexSensorPins[NUM_FLEX] = {26, 27, 14, 12, 13}; // Flex sensor pins
const int hallSensorPins[NUM_HALL] = {34, 35, 32}; // Hall sensor pins
int hallSensorValues[NUM_HALL];                            // Store Hall sensor values
int flexSensorValues[NUM_FLEX];

void setup() {
  Serial.begin(115200);            // Initialize Serial communication
  if (BLUETOOTH) SerialBT.begin("ESP32", true); // Start Bluetooth with a device name
  delay(100);
  myPrintln("Bluetooth Device is Ready to Pair");
  
  for (int i = 0; i < NUM_FLEX; i++) {
    pinMode(flexSensorPins[i], INPUT); // Set Flex sensor pins as input
  }

  for (int i = 0; i < NUM_HALL; i++) {
    pinMode(hallSensorPins[i], INPUT);  // Set Hall sensor pins as input
  }

  myPrintln("Adafruit MPU6050 test!");

  // Initialize MPU6050
  if (!mpu.begin()) {
    myPrintln("Failed to find MPU6050 chip");
    while (1) delay(10);
  }
  myPrintln("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  myPrint("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
    case MPU6050_RANGE_2_G: myPrintln("+-2G"); break;
    case MPU6050_RANGE_4_G: myPrintln("+-4G"); break;
    case MPU6050_RANGE_8_G: myPrintln("+-8G"); break;
    case MPU6050_RANGE_16_G: myPrintln("+-16G"); break;
  }

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  myPrint("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
    case MPU6050_RANGE_250_DEG: myPrintln("+- 250 deg/s"); break;
    case MPU6050_RANGE_500_DEG: myPrintln("+- 500 deg/s"); break;
    case MPU6050_RANGE_1000_DEG: myPrintln("+- 1000 deg/s"); break;
    case MPU6050_RANGE_2000_DEG: myPrintln("+- 2000 deg/s"); break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  myPrintln("Filter bandwidth set to: 21 Hz");

  delay(100);
}

int getSensorData(int pin) {
  float sensorData = analogRead(pin);
  float sensorDataPercent = int(((sensorData - 2900) / (4095 - 2900)) * 100) / float(100);
  sensorDataPercent = (1 - sensorDataPercent) * 100;
  return sensorDataPercent;
}

/*void loop() {
  // Only send data if Bluetooth is connected
  if (SerialBT.connected()) {
    // Build one packet containing all data
    String packet = "";

    // Read Hall sensor values
    packet += "Analog Sensor Values: [";
    for (int i = 0; i < NUM_HALL; i++) {
      hallSensorValues[i] = analogRead(hallSensorPins[i]);
      packet += String(hallSensorValues[i]);
      if (i < NUM_HALL - 1) packet += ", ";
    }
    packet += "]\n";

    // Read MPU6050 data
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    packet += "Accel: " + String(a.acceleration.x) + ", " 
                      + String(a.acceleration.y) + ", " 
                      + String(a.acceleration.z) + "\n";
    packet += "Rot: " + String(g.gyro.x) + ", " 
                    + String(g.gyro.y) + ", " 
                    + String(g.gyro.z) + "\n";

    // Read Flex sensor values
    packet += "Flex Val: [";
    for (int i = 0; i < NUM_FLEX; i++) {
      flexSensorValues[i] = getSensorData(flexSensorPins[i]);
      packet += String(flexSensorValues[i]) + "%";
      if (i < NUM_FLEX - 1) packet += ", ";
    }
    packet += "]\n";

    // Send the entire packet over Bluetooth
    SerialBT.print(packet);
    SerialBT.flush();  // Ensure the data is pushed out
  }
  else {
    Serial.print("Disconnected");
  }

  delay(250); // Increase delay to give Windows time to process data
}*/

void loop() {
  //Serial.print("In loop");
  // Print sensor data regardless of BT connection.
  for (int i = 0; i < NUM_HALL; i++) {
    hallSensorValues[i] = analogRead(hallSensorPins[i]);
    myPrint(String(hallSensorValues[i]));
    if (i < NUM_HALL) myPrint(", ");
  }

  for (int i = 0; i < NUM_FLEX; i++) {
    flexSensorValues[i] = analogRead(flexSensorPins[i]);
    myPrint(String(flexSensorValues[i]));
    if (i < NUM_FLEX) myPrint(", ");
  }

  // Read MPU6050 data
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  myPrint(String(a.acceleration.x));
  myPrint(", ");
  myPrint(String(a.acceleration.y));
  myPrint(", ");
  myPrint(String(a.acceleration.z));
  myPrint(", ");
  myPrint(String(g.gyro.x));
  myPrint(", ");
  myPrint(String(g.gyro.y));
  myPrint(", ");
  myPrint(String(g.gyro.z));
  myPrintln();

  // Only check Bluetooth status if Bluetooth is enabled
  if (BLUETOOTH) {
    if (!SerialBT.connected()) {
      Serial.print("Bluetooth not connected\n");
    }
  }

  delay(50); // Adjust delay as needed
}