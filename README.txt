## Data files

- "LOG00075_NormalHandling.BFL.csv" Carrying and handling of the drone
- "LOG00172_ThrowAndFlight1.BFL.csv"
- "LOG00207_ThrowAndFlight2.BFL.csv"
- "LOG00208_ThrowAndFlight3.BFL.csv"
- "LOG00211_ThrowsLab.BFL.csv" Throws and catching


## Fields

- gyroADC[0-2] --> gyro output in deg/s, filtered with 75Hz First Order Lowpass filter. 
- accSmooth[0-2] --> accelerometer output in 2048*1g, filtered with 25Hz First Order Lowpass filter.
- motor[0-3] --> command sent to the motors (between 0 and 1000)
- quat[0-3] --> Attitude quaternion * 8127. Computed onboard. [w x y z] order

