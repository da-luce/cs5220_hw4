#!/usr/bin/env bash

set -e


# Show usage if help is requested
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
  echo "Usage: $0 [sizeX] [sizeY] [M] [H] [N] [--debug|--debug-instr]"
  echo "  sizeX: Size X (default: 2)"
  echo "  sizeY: Size Y (default: 2)"
  echo "  M: M dimension (default: 2)"
  echo "  H: H dimension (default: 2)"
  echo "  N: N dimension (default: 2)"
  echo "  --debug: Enable debug mode (landing,router)"
  echo "  --debug-instr: Enable instruction trace debug mode"
  exit 0
fi

# Parse positional parameters with defaults
sizeX=${1:-2}
sizeY=${2:-2}
M=${3:-4}
H=${4:-8}
N=${5:-4}

# Parse debug flags from remaining arguments
for arg in "$@"; do
  if [[ "$arg" == "--debug" ]]; then
    export APPTAINERENV_SIMFABRIC_DEBUG=landing,router
    export SIMFABRIC_DEBUG=landing,router
  elif [[ "$arg" == "--debug-instr" ]]; then
    export APPTAINERENV_SIMFABRIC_DEBUG=landing,router,inst_trace
    export SIMFABRIC_DEBUG=landing,router,inst_trace
  fi
done

./clean.sh

echo "Running with configuration: sizeX=$sizeX, sizeY=$sizeY, M=$M, H=$H, N=$N"

cslc --arch=wse2 ./layout.csl --fabric-dims=$(( (2 * 4) + $sizeX )),$(( (2 * 1) + $sizeY )) \
--fabric-offsets=4,1 --params=kernel_x_dim:${sizeX},kernel_y_dim:${sizeY},M:${M},N:${N},H:${H} \
--params=MEMCPYH2D_DATA_1_ID:2 \
--params=MEMCPYH2D_DATA_2_ID:3 \
--params=MEMCPYD2H_DATA_1_ID:4 \
--max-inlined-iterations 150 \
-o out --memcpy --channels 1

cs_python run.py --name out
