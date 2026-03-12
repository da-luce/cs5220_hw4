#!/usr/bin/env bash

set -e

# Test configurations: sizeX sizeY M H N
# Constraints:
#   M % sizeY == 0
#   H % sizeX == 0
#   N % sizeX == 0
#   N % sizeY == 0
#   sizeX >= 2, sizeY >= 2
configs=(
  # Square grids, power-of-2
  "2 2  8  8  8"
  "4 4 16 16 16"

  # Asymmetric grids (sizeY > sizeX), power-of-2
  "2 4  8  8  8"
  "2 8 16 16 16"
  "4 8 16 16 16"
  "4 8 32 16 32"

  # Larger matrices on small grid (big per-PE work)
  "2 2 16 16 16"
  "2 4 16  8 16"

  # Odd/non-power-of-2 grid dimensions (tests PSUM with 1 middle PE)
  "3 3  9  9  9"
  "3 6  6  9  6"
  "3 6 12  9 18"

  # Large kernel_x_dim (deep PSUM chain: 4+ middle PEs)
  "6 6 12 12 12"

  # Minimum per-PE work: dM=1, dH=1, dN_y=1
  "2 2  2  2  2"
  "4 4  4  4  4"
  "3 3  3  3  3"

  # Large dN_y relative to dH (many B-columns, short hidden dim)
  "2 4  4  4 16"
  "2 2  4  2 16"

  # Large dH relative to dN_y (long hidden dim, few B-columns)
  "2 2  4 16  2"
  "4 4  4 16  4"

  # High asymmetry ratio (sizeY/sizeX = 4)
  "2 8  8  8  8"
  "2 8  8  4 16"

  # sizeX > sizeY
  "4 2  4  4  4"
  "4 2  8  8  8"
  "8 2 16  8 16"
  "4 3  6  8 12"
  "6 2  4  6  6"
  "6 3  6  6  6"
  "8 4  8  8  8"

  # sizeX == sizeY
  "3 3  6  6  6"
)

PASS=0
FAIL=0
ERRORS=()

for cfg in "${configs[@]}"; do
  read -r sizeX sizeY M H N <<< "$cfg"
  echo "========================================"
  echo "Testing sizeX=$sizeX sizeY=$sizeY M=$M H=$H N=$N"
  echo "========================================"

  if ! ./run.sh "$sizeX" "$sizeY" "$M" "$H" "$N"; then
    echo "FAIL: sizeX=$sizeX sizeY=$sizeY M=$M H=$H N=$N"
    FAIL=$((FAIL + 1))
    ERRORS+=("sizeX=$sizeX sizeY=$sizeY M=$M H=$H N=$N")
    continue
  fi

  echo "PASS: sizeX=$sizeX sizeY=$sizeY M=$M H=$H N=$N"
  PASS=$((PASS + 1))
done

echo ""
echo "========================================"
echo "Results: $PASS passed, $FAIL failed out of ${#configs[@]} configs"
echo "========================================"
if [[ ${#ERRORS[@]} -gt 0 ]]; then
  echo "Failed configs:"
  for err in "${ERRORS[@]}"; do
    echo "  $err"
  done
  exit 1
fi
