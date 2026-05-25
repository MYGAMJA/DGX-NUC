<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/releases
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/releases.md
synced: 2026-05-24
title: Releases (다운로드 허브)
-->

# Releases

{% hint style="info" %}
Last update: 2025-09-07
{% endhint %}

### Bill Of Materials (BOM)

{% content-ref url="/pages/JwIYLwqBtQ8gg77xA9vS" %}
[Materials and Parts (BOM)](/docs/getting-started-with-hardware/materials-and-parts-bom.md)
{% endcontent-ref %}

### CAD Models

| Berkeley Humanoid Lite | [Onshape Link](https://cad.onshape.com/documents/fc6443b1d89dcba950e85b60/w/94a26479973a4098a5884426/e/be3fe37849edbeadc95b9bf8?configuration=default\&renderMode=0\&uiState=67d7e630bb752737e88992d7)                       |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6512 Actuator          | [Onshape Link](https://cad.onshape.com/documents/55ab471d620553f44eac2d08/w/ae94825f64e460a99fce5bb9/e/087203429f83a08c02d31a2b?configuration=List_V0PVm1ztiuoOfK%3DDefault\&renderMode=0\&uiState=67d7e6a395a3c112cb3ba826) |
| 5010 Actuator          | [Onshape Link](https://cad.onshape.com/documents/192ab9c484f00d0dd33b8f01/w/69525ef69bcb5d7c0991e587/e/cc160d2ca1454a9219f579b3?renderMode=0\&uiState=67d7e5a3bb752737e8899291)                                              |
| USB-CAN Adapter Case   | [Onshape Link](https://cad.onshape.com/documents/dc5bf30cc5a66c2be4659344/w/30c67813e1458487202ea2b9/e/4f46e96306414e60771b645a?renderMode=0\&uiState=6860dc70ad1e0048620ab677)                                              |

### 3D Printing Project Files

| Berkeley Humanoid Lite | [Bambu Lab MakerWorld Link](https://makerworld.com/en/models/1327260-berkeley-humanoid-lite#profileId-1364871)       |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 6512 Actuator          | [Bambu Lab MakerWorld Link](https://makerworld.com/en/models/1220823-6512-cycloidal-gear-actuator#profileId-1364989) |
| 5010 Actuator          | [Bambu Lab MakerWorld Link](https://makerworld.com/en/models/1279205-5010-cycloidal-gear-actuator#profileId-1364911) |

### Software Repositories

#### Main Project Repository

{% embed url="<https://github.com/HybridRobotics/Berkeley-Humanoid-Lite>" %}

#### Robot Description Assets (URDF, MJCF, USD)

{% embed url="<https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Assets>" %}

#### Low-level Control Code

{% embed url="<https://github.com/HybridRobotics/Berkeley-Humanoid-Lite-Lowlevel>" %}

#### Firmware

{% embed url="<https://github.com/T-K-233/Recoil-Motor-Controller-BESC>" %}

#### Windows SteamVR Code for Teleoperation

{% embed url="<https://github.com/ucb-bar/SteamVR-Bridge>" %}

### Release Log

**2025-11-10** Fix incorrect joint limit values in "Joint ID Mapping" page.

**2025-09-07** Fix missing Python locomotion instruction.

**2025-09-03** Update robot description file generation instructions.

**2025-08-23** Fix 3D print file on MakeWorld, replacing the wrong 5010-5010 housing part and add the missing limit stopper part; add instruction on mounting the magnet on motor shaft; update instruction for 3D printing.

**2025-07-07** Fix motor calibration procedure description, add note for rotation direction.

**2025-07-05** Add video instruction on ESC firmware configuration, motor calibration, and a demo code driving a single motor.

**2025-06-28** Add more wiring instructions, IMU upgrade, and teleoperation instructions; add USB-CAN Adapter Onshape link.

**2025-06-26** Add teleoperation code.

**2025-06-13** Add instructions to set up environment with uv.

**2025-06-09** Add robot wiring diagram, add instructions on connecting cables.

**2025-05-29** Add missing 6803 bearing and M2 insert in the humanoid tab in BOM, update M4 hex standoff links.

**2025-05-23** Fix incorrect bearing description in BOM (swapped MR106-2RS and 6701-2RS).

**2025-05-21** Fix MakerWorld .3mf files of the 6512 actuator.

**2025-04-28** Initial release.


---
