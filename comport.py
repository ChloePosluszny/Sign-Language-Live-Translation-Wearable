import serial
import time

# Configure these values
BLUETOOTH_COM_PORT = 'COM5'  # Find this in your OS
BAUD_RATE = 115200           # Must match ESP32's baud rate

def read_bluetooth_serial():
    try:
        with serial.Serial(BLUETOOTH_COM_PORT, BAUD_RATE) as bt_serial:
            print(f"Connected to {BLUETOOTH_COM_PORT}")
            consecutive_failures = 0
            while True:
                try:
                    data = bt_serial.readline().decode('utf-8', errors='ignore').strip()
                    if data:
                        print(data)
                        consecutive_failures = 0  # Reset failure counter
                    else:
                        consecutive_failures += 1  # Count empty responses
                        
                        if consecutive_failures > 5:
                            print("Warning: No data received for 5 cycles. Possible Bluetooth drop.")
                            consecutive_failures = 0
                except serial.SerialException:
                    print("Error: Serial exception occurred.")
                    break
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if 'bt_serial' in locals() and bt_serial.is_open:
            bt_serial.close()
            print("Port closed")

if __name__ == "__main__":
    read_bluetooth_serial()