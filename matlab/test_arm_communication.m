clc; clear all;
arduino = serialport('COM8', 115200);
pause(2);
writeline(arduino, "0:10,1:10")