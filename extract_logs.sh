#!/bin/bash

# Parse command line arguments with defaults
off_x=4
off_y=1
X=${1:-4}
Y=${2:-4}


# Show usage if help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  echo "Usage: $0 [X] [Y]"
  echo "  X: Width (default: 4)"
  echo "  Y: Height (default: 4)"
  exit 0
fi

echo "Cleaning up old logs..."
rm sim*.log

echo "Extracting logs for rect region of ($off_x,$off_y) to ($((off_x + X - 1)),$((off_y + Y - 1)))..."

for i in $(seq 1 $Y); do 
for j in $(seq 1 $X); do 
  x=$(( off_x - 1 + j ))
  y=$(( off_y - 1 + i ))
  grep "P$x\.$y" sim.log > "sim${x}_${y}.log"
  grep "Loc" "sim${x}_${y}.log" > "sim_sprnt${x}_${y}.log";
done
done

echo "SUCCESS"
