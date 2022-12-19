# MetaMotion-Gait-Tracking

In this project, there is a code to colect the data from a MetaMotionRL, witch uses a BMI160 acelerometer and a BMI160 gyroscope too.
In the end of the code we use the Fusion AHRS algorithm to do the gait tracking and plot a 2D to show the path taken by the sensor.

## Prerequisites


To use this code you need to install mbientlab to use the metamotions data colect functions and install fusion library

Follow the instructions on this github to install Metawear-SDK:
[MetaWear-SDK](https://github.com/mbientlab/MetaWear-SDK-Python)

To install Fusion just paste this on terminal:

`pip install imufusion`
