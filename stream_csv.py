# usage: python3 MetaWearStream.py

from __future__ import print_function
from mbientlab.metawear import MetaWear, libmetawear, parse_value, create_voidp, create_voidp_int
from mbientlab.metawear.cbindings import *
from time import sleep
from threading import Event

import platform
import sys
import re
import os
import time

import threading
import pandas as pd

semaphore = threading.Semaphore(1) # the semaphore will control the function below


def stop_data():
# input something to stop the colect

	stop = input()

	semaphore.release()

t1 = threading.Thread(target=stop_data)

t1.start()


# -------------------------------------------------------------------------------
#
#	VARIABLES TO SETUP FOR CONNECTION :)
#
# -------------------------------------------------------------------------------


device_name = "MetaMotionRL" 
metamotion_mac_address = "E6:38:2A:F1:24:18"
bluetooth_adapter_mac_address = "3C:58:C2:F0:86:7D"
sampling_time = 10.0


if(metamotion_mac_address == " " or bluetooth_adapter_mac_address == " " or device_name == " "):
	print("You did not setup the script for connection. \nPlease open the .py file and compile the mac address variables!")
	exit()


if sys.version_info[0] == 2:
    range = xrange
    
# acc callback
def acc_data_handler(ctx, data):
	global d, accsamples, accFile, r
	#print("ACC: %s -> %s" % (d.address, parse_value(data)))
	axisValues = r.findall(str(parse_value(data)))  
	accFile.write("%d,%s,%s,%s\n" % (data.contents.epoch, axisValues[0],  axisValues[1],  axisValues[2]))
	accsamples += 1

# gyro callback
def gyro_data_handler(ctx, data):
	global d, gyrosamples, gyroFile
	#print("GYRO: %s -> %s" % (d.address, parse_value(data)))
	axisValues = r.findall(str(parse_value(data)))  
	gyroFile.write("%d,%s,%s,%s\n" % (data.contents.epoch, axisValues[0],  axisValues[1],  axisValues[2]))
	gyrosamples += 1

# mag callback
def mag_data_handler(ctx, data):
	global d, magsamples, magFile
	#print("MAG: %s -> %s" % (d.address, parse_value(data)))
	axisValues = r.findall(str(parse_value(data)))  
	magFile.write("%d,%s,%s,%s\n" % (data.contents.epoch, axisValues[0],  axisValues[1],  axisValues[2]))
	magsamples += 1

#temp callback
def temp_data_handler(ctx, data):
	global d, tempsamples, tempFile
	#print("TEMP: %s -> %s" % (d.address, parse_value(data)))
	tempFile.write("%d,%s\n" % (data.contents.epoch, parse_value(data)))
	tempsamples += 1

#pressure callback
def press_data_handler(ctx, data):
	global d, presssamples, pressFile
	#print("PRESS: %s -> %s" % (d.address, parse_value(data)))
	pressFile.write("%d,%s\n" % (data.contents.epoch, parse_value(data)))
	presssamples += 1

accsamples = 0
gyrosamples = 0
magsamples = 0
tempsamples = 0
presssamples = 0
accCallback = FnVoid_VoidP_DataP(acc_data_handler)
gyroCallback = FnVoid_VoidP_DataP(gyro_data_handler)
magCallback = FnVoid_VoidP_DataP(mag_data_handler)
tempCallback = FnVoid_VoidP_DataP(temp_data_handler)
pressCallback = FnVoid_VoidP_DataP(press_data_handler)

# create files
# filename = "output/acc_" + device_name + ".csv"

initial_time = str(int(time.time()))

os.makedirs(os.getcwd() + "/output/" + (initial_time), exist_ok=True)

filename = (os.getcwd() + "/output/" + initial_time + "/" + initial_time + "acc.csv")

accFile = open(filename, "w")
accFile.write("epoch,valueX,valueY,valueZ\n")

# filename = "output/gyro_" + device_name + ".csv"

filename = (os.getcwd() + "/output/" + initial_time + "/" + initial_time + "gyro.csv")

gyroFile = open(filename, "w")
gyroFile.write("epoch,valueX,valueY,valueZ\n")


# filename = "output/mag_" + device_name + ".csv"
# magFile = open(filename, "w")
# magFile.write("epoch,valueX,valueY,valueZ\n")

# define a regular expression to take axes from the output - will match all floats
r = re.compile("[+-]?[0-9]*[.][0-9]+")

# connect to the MetaWear sensor, using the address specified before

d = MetaWear(metamotion_mac_address, hci_mac = bluetooth_adapter_mac_address)
d.connect()
print("Connected to " + d.address + " over " + ("USB" if d.usb.is_connected else "BLE"))
e = Event()

# configure

#This API call configures, for the device in argument #1, min, max connection interval, latency and timeout.
libmetawear.mbl_mw_settings_set_connection_parameters(d.board, 7.5, 7.5, 0, 6000)
sleep(2.0)

# setup acc

#libmetawear.mbl_mw_acc_get_packed_acceleration_data_signal(d.board)

#Set output data rate for the Bosch sensor. Only some values are allowed and defined in the .h
libmetawear.mbl_mw_acc_bmi160_set_odr(d.board, AccBmi160Odr._100Hz) # BMI 270 specific call trocar para 160 ou generico NAO USAR GENERICO
#Set the range for the Bosch sensor. Default value to 0-4g.
libmetawear.mbl_mw_acc_bosch_set_range(d.board, AccBoschRange._16G)
#Applies ODR and Range to the sensor.
libmetawear.mbl_mw_acc_write_acceleration_config(d.board)

# setup gyro
#Same API calls as before.
libmetawear.mbl_mw_gyro_bmi160_set_range(d.board, GyroBoschRange._2000dps);
libmetawear.mbl_mw_gyro_bmi160_set_odr(d.board, GyroBoschOdr._100Hz);
libmetawear.mbl_mw_gyro_bmi160_write_config(d.board);
    
# setup mag
#Stop the magnetometer
#libmetawear.mbl_mw_mag_bmm150_stop(d.board)
#Sets the power mode to one of the recommended presets. High Accuracy is 20Hz and consumes a lot, REGULAR is 10Hz and is nice to the battery.acc
#libmetawear.mbl_mw_mag_bmm150_set_preset(d.board, MagBmm150Preset.HIGH_ACCURACY)


# get acc and subscribe
#Get the data signal representing acceleration data
acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(d.board)
#And subscribe, defining the callback function to be called.
libmetawear.mbl_mw_datasignal_subscribe(acc, None, accCallback)

# get gyro and subscribe
#Same as before.
gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(d.board)
libmetawear.mbl_mw_datasignal_subscribe(gyro, None, gyroCallback)

# get mag and subscribe
#mag = libmetawear.mbl_mw_mag_bmm150_get_b_field_data_signal(d.board)
#libmetawear.mbl_mw_datasignal_subscribe(mag, None, magCallback)
    
# start acc
libmetawear.mbl_mw_acc_enable_acceleration_sampling(d.board)
libmetawear.mbl_mw_acc_start(d.board)

# start gyro
libmetawear.mbl_mw_gyro_bmi160_enable_rotation_sampling(d.board)
libmetawear.mbl_mw_gyro_bmi160_start(d.board)

# start mag
#libmetawear.mbl_mw_mag_bmm150_enable_b_field_sampling(d.board)
#libmetawear.mbl_mw_mag_bmm150_start(d.board)

# sleep
#sleep(sampling_time) # remove the '#'' if you want to use time to limit the data colect



# stop

semaphore.acquire()

t1.join()


libmetawear.mbl_mw_gyro_bmi160_stop(d.board)
libmetawear.mbl_mw_gyro_bmi160_disable_rotation_sampling(d.board)

libmetawear.mbl_mw_acc_stop(d.board)
libmetawear.mbl_mw_acc_disable_acceleration_sampling(d.board)

#libmetawear.mbl_mw_mag_bmm150_stop(d.board)
#libmetawear.mbl_mw_mag_bmm150_disable_b_field_sampling(d.board)

# libmetawear.mbl_mw_baro_bosch_stop(d.board)

#mag = libmetawear.mbl_mw_mag_bmm150_get_b_field_data_signal(d.board)
#libmetawear.mbl_mw_datasignal_unsubscribe(mag)

acc = libmetawear.mbl_mw_acc_get_acceleration_data_signal(d.board)
libmetawear.mbl_mw_datasignal_unsubscribe(acc)

gyro = libmetawear.mbl_mw_gyro_bmi160_get_rotation_data_signal(d.board)
libmetawear.mbl_mw_datasignal_unsubscribe(gyro)


#remove timer, event and unsubscribe
# libmetawear.mbl_mw_timer_remove(timer)
# sleep(1.0)

libmetawear.mbl_mw_event_remove_all(d.board)
sleep(1.0)

# libmetawear.mbl_mw_datasignal_unsubscribe(signal)
# sleep(2.0)

libmetawear.mbl_mw_debug_disconnect(d.board)

# recap
print("Total Samples Received")
print("ACC -> %d" % (accsamples))
print("GYR -> %d" % (gyrosamples))
#print("MAG -> %d" % (magsamples))

gyroFile.close()
accFile.close()

    
# Copiar os samples de um arquivo para outro, colocar acelerometro do lado direito do gyro

print(os.getcwd())

df_acc = pd.read_csv(os.getcwd() + "/output/" + initial_time + "/" + initial_time + "acc.csv",sep=',')
df_gyro = pd.read_csv(os.getcwd() + "/output/" + initial_time + "/" + initial_time + "gyro.csv",sep=',')

milisec = [0]

epoch_ms = []


epoch_ms = list(df_gyro['epoch'])

milisec = [0] + [epoch_ms[i + 1] - epoch_ms[i] for i in range(len(epoch_ms) - 1)]
for i in range(1, len(milisec)):
	milisec[i] += milisec[i-1] 


df_gyro = df_gyro.rename(columns={'valueX': 'valueXgyro'})
df_gyro = df_gyro.rename(columns={'valueY': 'valueYgyro'})
df_gyro = df_gyro.rename(columns={'valueZ': 'valueZgyro'})

df_acc = df_acc.rename(columns={'valueX': 'valueXacc'})
df_acc = df_acc.rename(columns={'valueY': 'valueYacc'})
df_acc = df_acc.rename(columns={'valueZ': 'valueZacc'})

df_acc.drop(0,inplace=True)
df_acc.drop(len(df_gyro) - 1,inplace=True)


df_gyro = df_gyro.join(df_acc['valueXacc'], how='right')
df_gyro = df_gyro.join(df_acc['valueYacc'], how='right')
df_gyro = df_gyro.join(df_acc['valueZacc'], how='right')

milisec[:] = [x / 1000 for x in milisec]

#df_gyro.assign(epoch=milisec[:len(milisec)])
df_gyro['epoch'] = pd.Series(milisec)


# Reset index after drop
df_gyro=df_gyro.dropna().reset_index(drop=True)


df_gyro.to_csv(os.getcwd() + "/output/" + initial_time + "/" + initial_time + 'gyro_plus_acc.csv',index=False)




from dataclasses import dataclass
from matplotlib import animation
from scipy.interpolate import interp1d
import imufusion
import matplotlib.pyplot as pyplot
import numpy

# Import sensor data ("short_walk.csv" or "long_walk.csv")
data = numpy.genfromtxt(os.getcwd() + "/output/" + initial_time + "/" + initial_time + 'gyro_plus_acc.csv', delimiter=",", skip_header=1)

sample_rate = 100  # 400 Hz

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]

# Plot sensor data
# figure, axes = pyplot.subplots(nrows=6, sharex=True, gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]})

# figure.suptitle("Sensors data, Euler angles, and AHRS internal states")

# axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="Gyroscope X")
# axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Gyroscope Y")
# axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Gyroscope Z")
# axes[0].set_ylabel("Degrees/s")
# axes[0].grid()
# axes[0].legend()

# axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="Accelerometer X")
# axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Accelerometer Y")
# axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Accelerometer Z")
# axes[1].set_ylabel("g")
# axes[1].grid()
# axes[1].legend()

# Intantiate AHRS algorithms
offset = imufusion.Offset(sample_rate)
ahrs = imufusion.Ahrs()

ahrs.settings = imufusion.Settings(0.3,  # gain
                                   10,  # acceleration rejection
                                   0,  # magnetic rejection
                                   0 * sample_rate)  # rejection timeout = 5 seconds

# Process sensor data
delta_time = numpy.diff(timestamp, prepend=timestamp[0])

euler = numpy.empty((len(timestamp), 3))
internal_states = numpy.empty((len(timestamp), 3))
acceleration = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    gyroscope[index] = offset.update(gyroscope[index])

    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], delta_time[index])

    euler[index] = ahrs.quaternion.to_euler()

    ahrs_internal_states = ahrs.internal_states
    internal_states[index] = numpy.array([ahrs_internal_states.acceleration_error,
                                          ahrs_internal_states.accelerometer_ignored,
                                          ahrs_internal_states.acceleration_rejection_timer])

    acceleration[index] = 9.81 * ahrs.earth_acceleration  # convert g to m/s/s

# Plot Euler angles
# axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
# axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
# axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
# axes[2].set_ylabel("Degrees")
# axes[2].grid()
# axes[2].legend()

# # Plot internal states
# axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
# axes[3].set_ylabel("Degrees")
# axes[3].grid()
# axes[3].legend()

# axes[4].plot(timestamp, internal_states[:, 1], "tab:cyan", label="Accelerometer ignored")
# pyplot.sca(axes[4])
# pyplot.yticks([0, 1], ["False", "True"])
# axes[4].grid()
# axes[4].legend()

# axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration rejection timer")
# axes[5].set_xlabel("Seconds")
# axes[5].grid()
# axes[5].legend()

# # Plot acceleration
# _, axes = pyplot.subplots(nrows=4, sharex=True, gridspec_kw={"height_ratios": [6, 1, 6, 6]})

# axes[0].plot(timestamp, acceleration[:, 0], "tab:red", label="X")
# axes[0].plot(timestamp, acceleration[:, 1], "tab:green", label="Y")
# axes[0].plot(timestamp, acceleration[:, 2], "tab:blue", label="Z")
# axes[0].set_title("Acceleration")
# axes[0].set_ylabel("m/s/s")
# axes[0].grid()
# axes[0].legend()

# Identify moving periods
is_moving = numpy.empty(len(timestamp))

for index in range(len(timestamp)):
    is_moving[index] = numpy.sqrt(acceleration[index].dot(acceleration[index])) > 0.3  # threshold = 3 m/s/s

margin = int(0.1 * sample_rate)  # 100 ms

for index in range(len(timestamp) - margin):
    is_moving[index] = any(is_moving[index:(index + margin)])  # add leading margin

for index in range(len(timestamp) - 1, margin, -1):
    is_moving[index] = any(is_moving[(index - margin):index])  # add trailing margin

# # Plot moving periods
# axes[1].plot(timestamp, is_moving, "tab:cyan", label="Is moving")
# pyplot.sca(axes[1])
# pyplot.yticks([0, 1], ["False", "True"])
# axes[1].grid()
# axes[1].legend()

# Calculate velocity (includes integral drift)
velocity = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    if is_moving[index]:  # only integrate if moving
        velocity[index] = velocity[index - 1] + delta_time[index] * acceleration[index]

# Find start and stop indices of each moving period
is_moving_diff = numpy.diff(is_moving, append=is_moving[-1])


@dataclass
class IsMovingPeriod:
    start_index: int = -1
    stop_index: int = -1


is_moving_periods = []
is_moving_period = IsMovingPeriod()

for index in range(len(timestamp)):
    if is_moving_period.start_index == -1:
        if is_moving_diff[index] == 1:
            is_moving_period.start_index = index

    elif is_moving_period.stop_index == -1:
        if is_moving_diff[index] == -1:
            is_moving_period.stop_index = index
            is_moving_periods.append(is_moving_period)
            is_moving_period = IsMovingPeriod()

# Remove integral drift from velocity
velocity_drift = numpy.zeros((len(timestamp), 3))

for is_moving_period in is_moving_periods:
    start_index = is_moving_period.start_index
    stop_index = is_moving_period.stop_index

    t = [timestamp[start_index], timestamp[stop_index]]
    x = [velocity[start_index, 0], velocity[stop_index, 0]]
    y = [velocity[start_index, 1], velocity[stop_index, 1]]
    z = [velocity[start_index, 2], velocity[stop_index, 2]]

    t_new = timestamp[start_index:(stop_index + 1)]

    velocity_drift[start_index:(stop_index + 1), 0] = interp1d(t, x)(t_new)
    velocity_drift[start_index:(stop_index + 1), 1] = interp1d(t, y)(t_new)
    velocity_drift[start_index:(stop_index + 1), 2] = interp1d(t, z)(t_new)

velocity = velocity - velocity_drift

# Plot velocity
# axes[2].plot(timestamp, velocity[:, 0], "tab:red", label="X")
# axes[2].plot(timestamp, velocity[:, 1], "tab:green", label="Y")
# axes[2].plot(timestamp, velocity[:, 2], "tab:blue", label="Z")
# axes[2].set_title("Velocity")
# axes[2].set_ylabel("m/s")
# axes[2].grid()
# axes[2].legend()

# Calculate position
position = numpy.zeros((len(timestamp), 3))

for index in range(len(timestamp)):
    position[index] = position[index - 1] + delta_time[index] * velocity[index]

# Plot position
# axes[3].plot(timestamp, position[:, 0], "tab:red", label="X")
# axes[3].plot(timestamp, position[:, 1], "tab:green", label="Y")
# axes[3].plot(timestamp, position[:, 2], "tab:blue", label="Z")
# axes[3].set_title("Position")
# axes[3].set_xlabel("Seconds")
# axes[3].set_ylabel("m")
# axes[3].grid()
# axes[3].legend()

# Print error as distance between start and final positions
print("Error: " + "{:.3f}".format(numpy.sqrt(position[-1].dot(position[-1]))) + " m")

# pyplot.savefig(os.getcwd() + "/output/" + initial_time + "/" + initial_time + "resumo_dos_dados.jpg")

# pyplot.show()

# Create a 2D ScatterPlot describing the path

minimal = min(position[:, 0]) if min(position[:, 0]) < min(position[:, 1]) else min(position[:, 1])
maximum = max(position[:, 0]) if max(position[:, 0]) > max(position[:, 1]) else max(position[:, 1])

pyplot.scatter(position[:, 0], position[:, 1],label="Caminho percorrido")
pyplot.scatter(position[0,0],position[0,1],color="red",marker="X",label="Começo")
pyplot.scatter(position[len(position)-1,0],position[len(position)-1,1],color="#7aeb7a",marker="o",label="Fim")

pyplot.xlim(minimal*1.5 -1 , maximum*1.5 + 1)
pyplot.ylim(minimal*1.5 -1, maximum*1.5 + 1)
pyplot.title("Vista por cima do caminho percorrido pelo MetaMotion")
pyplot.xlabel("Coordenadas de X (m)")
pyplot.ylabel("Coordenadas de Y (m)")
pyplot.legend()
pyplot.grid()

pyplot.savefig(os.getcwd() + "/output/" + initial_time + "/" + initial_time + "caminho_2d.jpg")


pyplot.show()


# Create 3D animation (takes a long time, set to False to skip)
if False:
    figure = pyplot.figure(figsize=(10, 10))

    axes = pyplot.axes(projection="3d")
    axes.set_xlabel("m")
    axes.set_ylabel("m")
    axes.set_zlabel("m")

    x = []
    y = []
    z = []

    scatter = axes.scatter(x, y, z)

    fps = 30
    samples_per_frame = int(sample_rate / fps)

    def update(frame):
        index = frame * samples_per_frame

        axes.set_title("{:.3f}".format(timestamp[index]) + " s")

        x.append(position[index, 0])
        y.append(position[index, 1])
        z.append(position[index, 2])

        scatter._offsets3d = (x, y, z)

        if (min(x) != max(x)) and (min(y) != max(y)) and (min(z) != max(z)):
            axes.set_xlim3d(min(x), max(x))
            axes.set_ylim3d(min(y), max(y))
            axes.set_zlim3d(min(z), max(z))

            axes.set_box_aspect((numpy.ptp(x), numpy.ptp(y), numpy.ptp(z)))

        return scatter

    anim = animation.FuncAnimation(figure, update,
                                   frames=int(len(timestamp) / samples_per_frame),
                                   interval=1000 / fps,
                                   repeat=False)

    anim.save(os.getcwd() + "/output/" + initial_time + "/" + initial_time + "animation_3d.gif", writer=animation.PillowWriter(fps))
    