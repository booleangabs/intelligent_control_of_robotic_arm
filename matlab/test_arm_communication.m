clc; clear all;
arduino = serialport('COM7', 115200);
pause(5);
writeline(arduino, "0:10,1:80,2:80")