#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32

# =============================================================================
# Infrastructure (provided) — do NOT modify below this line
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

compile_data = dict()
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

kernel_x_dim = int(compile_data['params']['kernel_x_dim'])
kernel_y_dim = int(compile_data['params']['kernel_y_dim'])
N = int(compile_data['params']['N'])
H = int(compile_data['params']['H'])
M = int(compile_data['params']['M'])

assert M % kernel_y_dim == 0, "M must be divisible by kernel_y_dim"
assert H % kernel_x_dim == 0, "H must be divisible by kernel_x_dim"
assert N % kernel_y_dim == 0, "N must be divisible by kernel_y_dim"
assert N % kernel_x_dim == 0, "N must be divisible by kernel_x_dim"

def prepare_h2d(matrix, rows, cols, global_transpose, memory_transpose):
  """Chunk a (rows x cols) matrix for H2D transfer across the PE grid.

  Global layout (how blocks are distributed across PEs):
    global_transpose=False  =>  rows split along Y, cols split along X ("natural")
                                  PE(x, y) gets block(y, x) with shape (d_row, d_col)
                                  where d_row = rows/kernel_y_dim, d_col = cols/kernel_x_dim
    global_transpose=True   =>  rows split along X, cols split along Y ("transposed")
                                  PE(x, y) gets block(x, y) with shape (d_row, d_col)
                                  where d_row = rows/kernel_x_dim, d_col = cols/kernel_y_dim

  Memory layout (how each local block is stored on the PE):
    memory_transpose=False  =>  stored as (d_row, d_col) row-major
    memory_transpose=True   =>  stored as (d_col, d_row) row-major (the transpose)

  Returns: (prepared_flat_array, words_per_pe)
  """
  if not global_transpose:
    d_row = rows // kernel_y_dim
    d_col = cols // kernel_x_dim
    chunked = matrix.reshape(kernel_y_dim, d_row, kernel_x_dim, d_col)
    chunked = chunked.transpose(0, 2, 1, 3)  # (Ky, Kx, d_row, d_col)
  else:
    d_row = rows // kernel_x_dim
    d_col = cols // kernel_y_dim
    chunked = matrix.reshape(kernel_x_dim, d_row, kernel_y_dim, d_col)
    chunked = chunked.transpose(2, 0, 1, 3)  # (Ky, Kx, d_row, d_col)

  if memory_transpose:
    chunked = chunked.transpose(0, 1, 3, 2)  # (Ky, Kx, d_col, d_row)

  return chunked.ravel(), d_row * d_col


def reconstruct_d2h(data, rows, cols, global_transpose, memory_transpose):
  """Reconstruct a (rows x cols) matrix from D2H data read back from the PE grid.

  Reverses the layout transformations applied by prepare_h2d.

  Returns: numpy array of shape (rows, cols)
  """
  if not global_transpose:
    d_row = rows // kernel_y_dim
    d_col = cols // kernel_x_dim
  else:
    d_row = rows // kernel_x_dim
    d_col = cols // kernel_y_dim

  if memory_transpose:
    result = data.reshape(kernel_y_dim, kernel_x_dim, d_col, d_row)
    result = result.transpose(0, 1, 3, 2)  # undo memory transpose → (Ky, Kx, d_row, d_col)
  else:
    result = data.reshape(kernel_y_dim, kernel_x_dim, d_row, d_col)

  if not global_transpose:
    result = result.transpose(0, 2, 1, 3).reshape(rows, cols)
  else:
    result = result.transpose(1, 2, 0, 3).reshape(rows, cols)

  return result


# =============================================================================
# Student configuration — imported from config.py
# =============================================================================
from config import (
  A_GLOBAL_TRANSPOSE, A_MEMORY_TRANSPOSE,
  B_GLOBAL_TRANSPOSE, B_MEMORY_TRANSPOSE,
  C_GLOBAL_TRANSPOSE, C_MEMORY_TRANSPOSE,
)


# =============================================================================
# Infrastructure (provided) — do NOT modify below this line
# =============================================================================

# Per-PE dimensions (derived from your global layout choices)
def local_dims(mat_rows, mat_cols, global_transpose):
  """Return (d_row, d_col) for a single PE given the layout choice."""
  if not global_transpose:
    return mat_rows // kernel_y_dim, mat_cols // kernel_x_dim
  else:
    return mat_rows // kernel_x_dim, mat_cols // kernel_y_dim

dM_A, dH_A = local_dims(M, H, A_GLOBAL_TRANSPOSE)
dH_B, dN_B = local_dims(H, N, B_GLOBAL_TRANSPOSE)
dM_C, dN_C = local_dims(M, N, C_GLOBAL_TRANSPOSE)

# Construct random test matrices
A = np.random.rand(M, H).astype(np.float32)
B = np.random.rand(H, N).astype(np.float32)
C_expected = A @ B

# Prepare data for H2D transfer
A_prepared, A_wpe = prepare_h2d(A, M, H, A_GLOBAL_TRANSPOSE, A_MEMORY_TRANSPOSE)
B_prepared, B_wpe = prepare_h2d(B, H, N, B_GLOBAL_TRANSPOSE, B_MEMORY_TRANSPOSE)

# Launch simulation
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)
A_symbol = runner.get_id('A')
B_symbol = runner.get_id('B')
C_symbol = runner.get_id('C')

runner.load()
runner.run()

# Copy A and B to device
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, A_wpe,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)
runner.memcpy_h2d(B_symbol, B_prepared, 0, 0, kernel_x_dim, kernel_y_dim, B_wpe,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Launch the computation
runner.launch('broadcast_pe', nonblock=False)

# Read back C
C_raw = np.zeros(kernel_x_dim * kernel_y_dim * dM_C * dN_C, dtype=np.float32)
runner.memcpy_d2h(C_raw, C_symbol, 0, 0, kernel_x_dim, kernel_y_dim, dM_C * dN_C,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Read back TSC timestamp buffer (6 x u16 per PE: start[0..2], end[0..2])
time_buf_symbol = runner.get_id('time_buf_u16')
time_buf_raw = np.zeros(kernel_x_dim * kernel_y_dim * 6, dtype=np.uint32)
runner.memcpy_d2h(time_buf_raw, time_buf_symbol, 0, 0, kernel_x_dim, kernel_y_dim, 6,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_16BIT,
  nonblock=False)

runner.stop()

# Reconstruct global C from per-PE blocks
C_result = reconstruct_d2h(C_raw, M, N, C_GLOBAL_TRANSPOSE, C_MEMORY_TRANSPOSE)

# Verify
np.testing.assert_allclose(C_result, C_expected, rtol=1e-05, atol=1e-03)
print("SUCCESS: C = A @ B verified.")

# --- Compute TSC timer statistics ---
def make_u48(w):
  """Combine 3 x u16 words into a 48-bit timestamp."""
  return int(w[0]) + (int(w[1]) << 16) + (int(w[2]) << 32)

time_buf_u16 = time_buf_raw.astype(np.uint16).reshape(kernel_y_dim, kernel_x_dim, 6)

cycles = np.zeros((kernel_y_dim, kernel_x_dim), dtype=np.int64)
for py in range(kernel_y_dim):
  for px in range(kernel_x_dim):
    t_start = make_u48(time_buf_u16[py, px, 0:3])
    t_end   = make_u48(time_buf_u16[py, px, 3:6])
    cycles[py, px] = t_end - t_start

print(f"\nTSC timer (cycles per PE):")
print(f"  Min:  {cycles.min()}")
print(f"  Max:  {cycles.max()}")
print(f"  Mean: {cycles.mean():.1f}")
