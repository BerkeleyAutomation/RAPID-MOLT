'''
This is the main script, running on the raspberry pie to control the irrigation actuation via the solenoids.

It reads from:
- emitterPinout.csv
- irrigationSchedule.csv

Format in irrigation Schedule.csv:
<pinout>, <hour>, <minute>, <second>, irrigated today (1 is yes, 0 is no)

The script logs every opening of a solenoid to: irrigationLog.txt
'''

import csv
import numpy as np
from time import sleep
import datetime
import RPi.GPIO as GPIO

irrigationSchedule = np.zeros((20, 5))
GPIO_pinout = np.zeros((20, 2))
killAll = False

# path where the irrigation log is saved
path = '/home/pi/Desktop/'
desktop_path = '/home/pi/Desktop/'
# path to the irrigation file
irrigation_file = 'irrigationSchedule.csv'

# settings for the Wet Area Reference Surface:
# Port:
wet_cloth_pot = 12
# the rate after which it should be irrigated in minutes
wet_cloth_rate = 30

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)


def OnShutdown():
    global killAll
    killAll = True


def openEmitter(emitterNum, duration):
    # GPIO pin 40 is the water pump
    global GPIO_pinout
    GPIO.setup(40, GPIO.OUT)
    sleep(1)
    GPIO.output(40, GPIO.LOW)

    GPIO.setup(int(GPIO_pinout[emitterNum, 1]), GPIO.OUT)
    print('emitter number ', str(emitterNum), ' is open')
    GPIO.output(int(GPIO_pinout[emitterNum, 1]), GPIO.LOW)
    sleep(duration)
    GPIO.output(int(GPIO_pinout[emitterNum, 1]), GPIO.HIGH)

    GPIO.output(40, GPIO.HIGH)
    print('emitter number ', str(emitterNum), ' is close')

    # write to log file
    f = open(desktop_path+'irrigationLog.txt', 'a+')
    date = datetime.datetime.now().date()
    time = datetime.datetime.now().time()
    f.write(str(date) + ' ' + str(time) + ' ' + 'emitter num: ' + str(emitterNum) + ' '
            + 'duration: ' + str(duration) + '\r\n')
    f.close()


def main():
    # global irrigationSchedule
    global killAll
    global GPIO_pinout
    killAll = False

    # set the current date
    date = datetime.datetime.now().date()
    previous_day = (date - datetime.timedelta(days=1)).day
    last_time_cloth_watered = datetime.datetime.now() - datetime.timedelta(minutes=30)
    
    # read and update pinout
    from_txt = np.genfromtxt(path+'emitterPinout.csv', delimiter=',')
    for row in range(0, 20, 1):
        for col in range(0, 2, 1):
            GPIO_pinout[row, col] = int(from_txt[row, col])
    
    # closing all emitter
    for i in range(0, 20 ,1):
       GPIO.setup(int(GPIO_pinout[i, 1]), GPIO.OUT)
       GPIO.output(int(GPIO_pinout[i, 1]), GPIO.HIGH)
    sleep(2)

    # constant loop
    while ~killAll:
        # update irrigationSchedule
        from_txt = np.genfromtxt(path+irrigation_file, delimiter=',')
        for row in range(0, 20, 1):
            for col in range(0, 5, 1):
                irrigationSchedule[row, col] = from_txt[row, col]

        # check if the day changed, if yes updated irrigation status
        date = datetime.datetime.now().date()
        if date.day != previous_day:
            print('day changed, initializing irrigation status...')
            # initializing the irrigation tracking
            for row in irrigationSchedule:
                irrigationSchedule[int(row[0]), 4] = 0
                with open(path+irrigation_file, 'wb') as cvsfile:
                    to_txt = csv.writer(cvsfile, delimiter=',')
                    for row in irrigationSchedule:
                        to_txt.writerow(row)
            previous_day = date.day

        # check if the wet cloth should be irrigated again
        if datetime.datetime.now() >= (last_time_cloth_watered + datetime.timedelta(minutes=wet_cloth_rate)):
                # set the irrigation schedule at wet cloth pot to "not irrigated"
                irrigationSchedule[wet_cloth_pot, 4] = 0
                # Write it to the CSV file
                with open(path+irrigation_file, 'wb') as cvsfile:
                    to_txt = csv.writer(cvsfile, delimiter=',')
                    for row in irrigationSchedule:
                        to_txt.writerow(row)
                # set last_time_cloth_watered to current
                last_time_cloth_watered = datetime.datetime.now()

        time_now = datetime.datetime.now().time()
        time_now_int = int(time_now.hour * 60 + time_now.minute)

        # iterate over the schedule and open corresponding emitters
        for row in irrigationSchedule:
            emitter_num = int(row[0])
            irrigation_hour = row[1]
            irrigation_minute = row[2]
            irrigation_duration = row[3]
            was_irrigated = row[4]

            # check which emitters should be open and open them
            if time_now_int > (irrigation_hour*60+irrigation_minute) and was_irrigated == 0:
                irrigationSchedule[emitter_num, 4] = 1
                with open(path + irrigation_file, 'wb') as cvsfile:
                    to_txt = csv.writer(cvsfile, delimiter=',')
                    for row in irrigationSchedule:
                        to_txt.writerow(row)
                openEmitter(emitter_num, irrigation_duration)

        sleep(2)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        OnShutdown()
