import serial
import time

# Configure these values
BLUETOOTH_COM_PORT = 'COM5'  # Ensure this is the correct port
BAUD_RATE = 115200           # Must match ESP32's baud rate

def read_bluetooth_serial():
    try:
        with serial.Serial(BLUETOOTH_COM_PORT, BAUD_RATE, timeout=1) as bt_serial:
            print(f"Connected to {BLUETOOTH_COM_PORT}")
            buffer = ''
            
            while True:
                # Read all available data and add to buffer
                data = bt_serial.read_all().decode('utf-8', errors='ignore')
                if data:
                    buffer += data
                    # Split buffer into lines
                    while '\n' in buffer:
                        line_end = buffer.find('\n')
                        line = buffer[:line_end].strip()
                        buffer = buffer[line_end + 1:]
                        if line:  # Ignore empty lines
                            print(line)
                else:
                    # Optional: Add a check for prolonged disconnection
                    pass
                    
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        print("Port closed")

if __name__ == "__main__":
    read_bluetooth_serial()