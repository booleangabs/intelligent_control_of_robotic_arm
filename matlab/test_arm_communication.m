clc; clear all;
arduino = serialport('COM7', 115200);

writeline(arduino, "0:10,1:10")