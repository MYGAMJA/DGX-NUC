#!/usr/bin/env bash
# BHL CAN 인터페이스 셋업 — udev 고정 이름(/dev/canable-left, -right) 사용.
# 실행: sudo bash /home/laba/DGX-NUC/BHL/can_up.sh
set -e

killall slcand 2>/dev/null || true
sleep 1

if [ ! -e /dev/canable-left ] || [ ! -e /dev/canable-right ]; then
    echo "ERROR: /dev/canable-left 또는 /dev/canable-right 가 없음."
    echo "  확인 1) udev 규칙 설치됐나: ls /etc/udev/rules.d/99-canable.rules"
    echo "  확인 2) CANable 2개 USB 연결됐나"
    ls -l /dev/canable-* 2>/dev/null || true
    exit 1
fi

slcand -o -c -s8 /dev/canable-left  can0
slcand -o -c -s8 /dev/canable-right can1
sleep 1
ip link set up can0
ip link set up can1

echo "=== CAN 인터페이스 ==="
ip -brief link show can0
ip -brief link show can1
echo "OK — can0=왼다리, can1=오른다리"
