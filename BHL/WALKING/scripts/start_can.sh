#!/bin/bash
# Hylion: 다리 전용 can0(왼), can1(오른)만 사용
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
