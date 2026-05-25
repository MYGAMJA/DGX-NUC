<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/motor-controller-firmware-execution-timing-information
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/motor-controller-firmware-execution-timing-information.md
synced: 2026-05-24
title: 펌웨어 타이밍
-->

# Motor Controller Firmware Execution Timing Information

The high dynamic nature of the motor drive circuit imposes a strict requirement on the realtimeness of the firmware. We must ensure that we can run everything within a single FOC commutation loop.

We performed several measurements on the Recoil firmware code running on the motor controller to profile the execution time.

The timing information is measured with a Siglent SDS 1202X-E oscilloscope via a GPIO header on the motor controller board.

The system clock is configured as 160 MHz. The compiler optimization is set to `-O2`.

## Current Control

Entire loop

### Encoder reading

<figure><img src="/files/W9M7tXFfyXLJ6U6rn6zV" alt=""><figcaption></figcaption></figure>

The delta time between issuing **HAL\_I2C\_Master\_Receive\_IT** and receiving the interrupt is 78 us.

Hence, the maximum frequency it can run at is 12 kHz.

### Clarke transform

<figure><img src="/files/NAkWbWLPyxDSSZkWhvie" alt=""><figcaption></figcaption></figure>

### Park transform

<figure><img src="/files/9FZLESaqZmNRLEcWzZWV" alt=""><figcaption></figcaption></figure>

### V\_D V\_Q target calculation

<figure><img src="/files/CyvEhsQnbLD56SfJobMO" alt=""><figcaption></figcaption></figure>

### Overmodulation

<figure><img src="/files/Pa2cx2GufjiM8pruzNcq" alt=""><figcaption></figcaption></figure>

### Inverse park transformation

<figure><img src="/files/nt1AF7eeuL31rJvzSC9G" alt=""><figcaption></figcaption></figure>

### SVPWM

<figure><img src="/files/mERBpc5ws5TOUGl0M2P7" alt=""><figcaption></figcaption></figure>


---
