<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/imu-comparision
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/in-depth-contents/imu-comparision.md
synced: 2026-05-24
title: IMU 비교 BNO085 vs IM10A
-->

# IMU Comparision

In this page we list the performance characteristics of the BNO085 IMU, which we originally used, and the IM10A IMU, which is the upgraded one that already provides USB interface.

| Parameter                                         | Unit          | BNO085    | IM10A         |
| ------------------------------------------------- | ------------- | --------- | ------------- |
| <mark style="color:red;">**Accelerometer**</mark> |               |           |               |
| Range                                             | g             | ±16       | ± 16          |
| Resolution                                        | mg / LSB      | 1         | 0.5           |
| RMS Noise                                         | mg            | 0.16      | 0.75 \~ 1     |
| Static Zero Drift                                 | mg            | ± 150     | ± 20 \~ 40    |
| Bandwidth                                         | Hz            | 8 \~ 1000 | 5 \~ 256      |
| <mark style="color:green;">**Gyroscope**</mark>   |               |           |               |
| Range                                             | °/s           | ± 2000    | ± 2000        |
| Resolution                                        | (°/s) / (LSB) | 0.0625    | 0.061         |
| RMS Noise                                         | °/s           | 0.014     | 0.028 \~ 0.07 |
| Static Zero Drift                                 | °/s           | ± 1       | ± 0.5 \~ 1    |
| Bandwidth                                         | Hz            | 12 \~ 523 | 5 \~ 256      |
| <mark style="color:blue;">**Magnetometer**</mark> |               |           |               |
| Range                                             | Gauss         | ± 13      | ± 2           |
| Resolution                                        | Gauss / LSB   | 0.003     | 0.0667        |


---
