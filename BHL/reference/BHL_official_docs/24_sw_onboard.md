<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/the-on-board-computer
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/the-on-board-computer.md
synced: 2026-05-24
title: 온보드 NUC
-->

# The On-board Computer

We are almost done!

The last step before running policy on the real robot is to set up the environment on the on-board NUC computer.

## Install Ubuntu 22.04

We find that Ubuntu 22.04 works well with our software and hardware stack on the robot. Follow [Ubuntu tutorial](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview) to install Ubuntu 22.04 on the NUC computer.

{% hint style="info" %}

## Note

To enter BIOS on the BeeLink N95 NUC computer, press XXXX during bootup.
{% endhint %}

## Install Dependencies

```
sudo apt install build-essential cmake net-tools can-utils python3-pip
```

## Test connection with joint actuators

As the first step, we need to initialize the CAN transport on the Linux side. To do this, we have prepared a script:

```bash
source ./scripts/start_can_transports.sh
```

{% hint style="info" %}

#### Note

Naturally, to stop the transports, you can do:

```bash
source ./scripts/stop_can_transports.sh
```

{% endhint %}

To test connection to individual actuator, run this command.

```bash
python ./berkeley_humanoid_lite_lowlevel/motor/ping.py --port can0 --id 1
```

`--port` corresponds to the port of the CAN device on linux. With the two arms and legs all connected, it should be `can[0,1,2,3]`

`--id`  corresponds to the CAN ID of the device. It should be in range `[1..14]`.

Alternatively, to test connection to all the motors, run this script.

```bash
python ./berkeley_humanoid_lite_lowlevel/robot/anyonehere.py
```

## Test connection with IMU

To test connection to the IMU, run this command.

```bash
python ./berkeley_humanoid_lite_lowlevel/robot/test_imu.py
```

## Test connection with Joystick

Run this script to launch the Python program that reads the joystick and broadcast the reading on UDP port.

```bash
python ./berkeley_humanoid_lite_lowlevel/policy/udp_joystick.py
```

## Run everything together

There are two ways to run the lowlevel code: C codebase and Python codebase.

{% hint style="info" %}

## Note

Before running the program, make sure that the CAN transports are started correctly:

```bash
source ./scripts/start_can_transports.sh
```

{% endhint %}

### C codebase

For locomotion tasks, we recommend to use the C codebase for better realtime gaurantee. The C codebase is under `./csrc/` directory.

On the on-board computer, do the following:

```bash
make
./build/real_humanoid
```

{% hint style="info" %}

## Real-time kernel performance

If running into performance issues where the USB-CAN adapter cannot maintain the required communication frequency, we can use the <https://xanmod.org/> real-time kernel instead.
{% endhint %}

### Python codebase

Go to the `./source/berkeley-humanoid-lite-lowlevel/` directory, and then run

```bash
uv run ./scripts/python/run_locomotion.py
```

The robot will follow the same state transition as in the Mujoco sim2sim simulation.

To visualize the robot, run

```bash
uv run ./scripts/sim2real/visualize.py
```


---
