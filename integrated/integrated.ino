#define MAX_PACKET_SIZE 64
#define NUM_HALL 3
#define NUM_FLEX 5
//#define LEFT_GLOVE
#define RIGHT_GLOVE
#define DELAY 50

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

  mpu.setGyroRange(MPU6050_RANGE_500_DEG);

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  delay(100);
}

void loop() {
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
  myPrint(", ");
  #ifdef LEFT_GLOVE
    myPrint("0");
  #endif
  #ifdef RIGHT_GLOVE
    myPrint("1");
  #endif
  myPrintln();

  // Only check Bluetooth status if Bluetooth is enabled
  if (BLUETOOTH) {
    if (!SerialBT.connected()) {
      Serial.print("Bluetooth not connected\n");
    }
  }

  delay(DELAY); // Adjust delay as needed
}