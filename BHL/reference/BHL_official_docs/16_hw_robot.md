<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/building-the-robot
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-hardware/building-the-robot.md
synced: 2026-05-24
title: 로봇 조립/배선
-->

# Building the Robot

Please follow these video tutorials to assemble the robot.

## Arm

{% embed url="<https://youtu.be/zsb3M3H1sr4>" %}

## Leg

{% embed url="<https://youtu.be/aRtWpbteiNA>" %}

## Entire robot

{% embed url="<https://youtu.be/SIGD8I-hwG8>" %}

## Wiring

After building the mechanical structure of the robot, connect the electrical components following this wiring diagram:

<figure><img src="/files/fyjgp8KGCkREGmRzcskY" alt=""><figcaption></figcaption></figure>

### Connecting CAN bus to the USB-CAN Adapter

The cables can be directly attache to the screw terminal on the USB-CAN adapter board. The ordering is CAN-L, CAN-H, GND. The signal names are also labeled at the back side of the PCB.

<figure><img src="/files/7sUpgyZwIpkzKIcTP4hh" alt=""><figcaption></figcaption></figure>

<figure><img src="/files/zrAAjt4NbBFNAiF2VeGe" alt=""><figcaption><p>Photo of the connection without the 3D printed case for better clarity</p></figcaption></figure>

### Joining the cables together

There are multiple ways to join the signal and power cables together. We provide our recommended ways for you reference.

For signal cables, we recommend directly solder them together and protect the solder joints with heat shrink tubes.

<figure><img src="/files/XEZCrLOV0V35QNK7Tbzc" alt=""><figcaption></figcaption></figure>

When first connecting power cables together, for easier debugging, the WAGO connectors can be used to quickly joining and detaching each actuators to the main power bus without soldering. They are available in multiple types, and we use both the two, three, and five ports on the robot.

<figure><img src="/files/a2XDNu5X5AAAm05WAYyl" alt=""><figcaption></figcaption></figure>

For a more permanant build, we recommend to solder the cables together directly. [This video](https://youtu.be/4xUBRMgcVhc?t=437\&si=RwDLI2K0Sdax4TTa) by Will Donaldson provides a good guide on how to solder these thicker cables. Between the actuators, the cables can be connected with XT30 and XT60 connectors. We use XT60 to connect the main cable together, with each actuator connected to this main power bus using XT30 connectors.&#x20;

<figure><img src="/files/mWSkLOeOPAL27Q8axUdF" alt=""><figcaption></figcaption></figure>

### IMU Connection

For the original version, we use an Arduino Nano to connect the IMU to the computer. Here are some photos of the connection for your reference.

<div><figure><img src="/files/0H5PC86mlbQXiNCa53Zy" alt=""><figcaption></figcaption></figure> <figure><img src="/files/8LThVE7sv2Fj3bJ2Zk2X" alt=""><figcaption></figcaption></figure> <figure><img src="/files/q64LsVivKb4rEQ2bGFXz" alt=""><figcaption></figcaption></figure></div>

We later found out this [IM10A IMU](https://www.hiwonder.com/products/imu-module?variant=40375875338327) that directly supports USB connection. Hence, we strongly recommend to use this IMU to avoid manually soldering the signal wires. The BOM is also updated to include this component.&#x20;

A detailed performance comparision between these two IMUs is available here:

{% content-ref url="/pages/2uqmCOTRXULM24M9aJNS" %}
[IMU Comparision](/docs/in-depth-contents/imu-comparision.md)
{% endcontent-ref %}


---
