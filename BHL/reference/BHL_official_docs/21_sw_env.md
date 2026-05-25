<!--
source: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/software-development-environment-overview
raw_md: https://berkeley-humanoid-lite.gitbook.io/docs/getting-started-with-software/software-development-environment-overview.md
synced: 2026-05-24
title: 개발환경
-->

# Software Development Environment Overview

## Directory walkthrough

The [HybridRobotics/Berkeley-Humanoid-Lite](https://github.com/hybridrobotics/berkeley-humanoid-lite) repository will be our working directory for everything.&#x20;

Inside the directory, there are three packages:

`source/berkeley_humanoid_lite/` contains the IsaacLab environment and task definitions.

`source/berkeley_humanoid_lite_assets/` contains robot descriptions (URDF, MJCF, and USD) and the script to export these description files from Onshape project.

`source/berkeley_humanoid_lite_lowlevel/` contains the lowlevel code running on the real robot. Only contents inside this folder is required to copy to the robot. We will cover that part in a later section.

## Code Editor

We recommend using [Cursor](https://www.cursor.com/en) / [VisualStudio Code](https://code.visualstudio.com/) as the editor.


---
