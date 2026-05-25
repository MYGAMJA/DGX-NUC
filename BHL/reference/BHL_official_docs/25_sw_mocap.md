<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/motion-capture-system
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/motion-capture-system.md
synced: 2026-05-24
title: SteamVR 텔레옵
-->

# Motion Capture System

To perform teleoperation control, we use the [SteamVR](https://partner.steamgames.com/vrlicensing) motion capture system to detect the user motion.

{% hint style="info" %}

### Note

Although it should be supported, we couldn't get the SteamVR working on Ubuntu. Here, we will be using another PC computer running Windows to drive the SteamVR setup. The motion capture data will be then streamed to the robot control system via UDP.

Please reach out if you have figured out how to set up the system entirely on Linux systems!
{% endhint %}

## Setting up the SteamVR tracking system

Please refer to the resource on [VIVE resources website](https://www.vive.com/us/support/vive-pro/) for detailed instructions on setting up the SteamVR tracking system.

During room calibration, make sure that the calibration arrow is pointed towards the +Y axis of your world coordinate frame. This ensures that the VR world coordinate aligns with our robot's coordinate frame.

## Setting up the motion capture codebase

On the SteamVR Windows computer, clone the SteamVR-Bridge repository:

```powershell
git clone https://github.com/ucb-bar/SteamVR-Bridge.git
```

We use uv for Python environment management. To install it on Windows, run the following command:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, create a new Python environment and install the required dependencies.

```powershell
uv venv --python 3.10
uv pip install -r requirements.txt
```

The VR Bridge will read the pose and button input information from the two Vive controllers, and stream the data to the Ubuntu robot computer via UDP communication.

Lastly, to establish the communication between the two computers, connect them together with a Ethernet cable, and set the IP address of the two computers with the following configuration:

| Computer                                   | IP Address |
| ------------------------------------------ | ---------- |
| Windows that connects to the SteamVR stuff | 172.28.0.8 |
| Ubuntu that connects to the robot          | 172.28.0.5 |

## Running the VR Bridge

To start the VR bridge program, simply run

```
uv run .\run_vr_bridge.py
```

{% hint style="info" %}

### Note

When running the script for the first time, it will automatically launch the SteamVR application. However, SteamVR will need to spend some time to detect and connect to the controllers. This might leads to the Python code unable to retrieve the controller. If this happens, re-run the Python code after SteamVR application is ready.
{% endhint %}

## Running the teleoperation program

To run the teleoperation program, run this command on the Ubuntu computer

```bash
uv run ./scripts/teleop/run_teleop.py
```

It will launch a MeshCat window that display coordination frame of:

* received VR controller pose
* robot end effector target pose
* current robot end effector pose


---
