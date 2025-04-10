import serial
import time


def run(string, port):
    arduino = serial.Serial(port=port, baudrate=115200, timeout=1)
    time.sleep(2)

    arduino.write(string)