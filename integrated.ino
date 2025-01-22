#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

int ADC1 = 26;
int ADC2 = 27;
int ADC3 = 14;
int ADC4 = 12;
int ADC5 = 13;
const int hallSensorPins[5] = {34, 35, 32, 33, 25};  // Pins connected to the Hall sensor outputs
int sensorValues[5];                               // Array to store sensor values

void setup() {
  Serial.begin(9600);                              // Initialize serial communication
  for (int i = 0; i < 5; i++) {
    pinMode(hallSensorPins[i], INPUT);             // Set each Hall sensor pin as input
  }
  while (!Serial)
    delay(10); // will pause Zero, Leonardo, etc until serial console opens

  Serial.println("Adafruit MPU6050 test!");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");

  pinMode(ADC1, INPUT);
  pinMode(ADC2, INPUT);
  pinMode(ADC3, INPUT);
  pinMode(ADC4, INPUT);
  pinMode(ADC5, INPUT);

  delay(100);
}

void loop() {
  // Read values from all Hall sensors
  for (int i = 0; i < 5; i++) {
    sensorValues[i] = analogRead(hallSensorPins[i]);  // Read and store sensor value
  }

  // Print all sensor values as an array
  Serial.print("Analog Sensor Values: [");
  for (int i = 0; i < 5; i++) {
    Serial.print(sensorValues[i]);
    if (i < 4) Serial.print(", ");                 // Add a comma between values
  }
  Serial.println("]");

  Serial.print("Digital Sensor Values: [");
  for (int i = 0; i < 5; i++) {
    // -1 for South Pole, 0 for no detected, 1 for North Pole
    if (sensorValues[i] >= 3950 || sensorValues[i] <= 1200) {
      Serial.print("1");
    } 
    else{
      Serial.print("0");
    }
    if (i < 4) Serial.print(", ");                 // Add a comma between values
  }
  Serial.print("]");

  Serial.println();                                // New line for the next output

  /* Get new sensor events with the readings */
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  /* Print out the values */
  Serial.print("Acceleration X: ");
  Serial.print(a.acceleration.x);
  Serial.print(", Y: ");
  Serial.print(a.acceleration.y);
  Serial.print(", Z: ");
  Serial.print(a.acceleration.z);
  Serial.println(" m/s^2");

  Serial.print("Rotation X: ");
  Serial.print(g.gyro.x);
  Serial.print(", Y: ");
  Serial.print(g.gyro.y);
  Serial.print(", Z: ");
  Serial.print(g.gyro.z);
  Serial.println(" rad/s");

  Serial.println("");

  //Thumb
  int sensorDataPercent = getSensorData(ADC1);

  Serial.println("Thumb Bent Percent: " + String(sensorDataPercent)+"%"); //4095 -> 2900
  Serial.println();

  //Index
  sensorDataPercent = getSensorData(ADC2);

  Serial.println("Index Bent Percent: " + String(sensorDataPercent)+"%"); //4095 -> 2900
  Serial.println();

  //Middle
  sensorDataPercent = getSensorData(ADC3);

  Serial.println("Middle Bent Percent: " + String(sensorDataPercent)+"%"); //4095 -> 2900
  Serial.println();

  //Ring
  sensorDataPercent = getSensorData(ADC4);

  Serial.println("Ring Bent Percent: " + String(sensorDataPercent)+"%"); //4095 -> 2900
  Serial.println();

  //Pinkie
  sensorDataPercent = getSensorData(ADC5);

  Serial.println("Pinkie Bent Percent: " + String(sensorDataPercent)+"%"); //4095 -> 2900
  Serial.println();

  delay(3000);                                     // Delay for 1 second
}

int getSensorData(int pin) {
  float sensorData = analogRead(pin);
  float sensorDataPercent =  int(((sensorData-2900)/(4095-2900)) * 100) / float(100);
  sensorDataPercent = (1-sensorDataPercent) * 100;
  return sensorDataPercent;
}