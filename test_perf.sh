#!/usr/bin/env bash

set -e

# Test configurations: sizeX sizeY M H N
# Constraints:
#   M % sizeY == 0
#   H % sizeX == 0
#   N % sizeX == 0
#   N % sizeY == 0
#   sizeX >= 2, sizeY >= 2
sizeX=8
sizeY=8
M=128
H=256
N=64
if ! output=$(./run.sh "$sizeX" "$sizeY" "$M" "$H" "$N"); then
  echo "FAIL: sizeX=$sizeX sizeY=$sizeY M=$M H=$H N=$N"
  exit 1
else
  echo "$output"
  echo "PASS: sizeX=$sizeX sizeY=$sizeY M=$M H=$H N=$N"
fi

# Parse TSC timer stats from stdout
tsc_min=$(echo "$output" | grep 'Min:' | grep -oE '[0-9]+')
tsc_max=$(echo "$output" | grep 'Max:' | grep -oE '[0-9]+')
tsc_mean=$(echo "$output" | grep 'Mean:' | grep -oE '[0-9]+(\.[0-9]+)?')

# Extract "cycles since a wavelet landed"
idle=$(grep 'cycles since a wavelet landed' sim.log | grep -oE '[0-9]+ cycles since a wavelet landed' | grep -oE '^[0-9]+')

# Extract total cycles
total=$(grep 'cycles=' sim.log | grep -oE 'cycles=[0-9]+' | grep -oE '[0-9]+')
echo "Total cycles: $total"
echo "Idle cycles:  $idle"
echo "Runtime:      $((total - idle))"
echo "TSC Min:      $tsc_min"
echo "TSC Max:      $tsc_max"
echo "TSC Mean:     $tsc_mean"
