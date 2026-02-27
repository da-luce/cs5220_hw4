#!/usr/bin/env cs_python

import argparse
import json
import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder # pylint: disable=no-name-in-module
from cerebras.sdk.sdk_utils import memcpy_view, input_array_to_u32

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="the test compile output dir")
# Your wont need this for the howmework, but you can specify the IP:port of your CS system if you want to run on hardware instead of sim.
parser.add_argument('--cmaddr', help="IP:port for CS system")
args = parser.parse_args()

# Get matrix dimensions from compile metadata, to avoid hardcoding them in the test
compile_data = dict()
with open(f"{args.name}/out.json", encoding='utf-8') as json_file:
  compile_data = json.load(json_file)

# PE grid dimensions
kernel_x_dim = int(compile_data['params']['kernel_x_dim'])
kernel_y_dim = int(compile_data['params']['kernel_y_dim'])

# Matrix dimensions, A \in R^{M,H}, B \in R^{H,N}, C \in R^{M,N}
N = int(compile_data['params']['N']) # columns
H = int(compile_data['params']['H']) # Hidden dimension, to be reduced
M = int(compile_data['params']['M']) # rows

# Approach: 
#  The physical matrix layout is is A^T \times B = C in memory and globally across 
#  the PEs A \times B = C, such that the hidden dimension is split and reduced across the kernel_x_dim. 
#  This means that each PE gets a chunk of the hidden dimension, and computes a partial reduction across that chunk, 
#  before the final results are combined to produce C.
#  We split N and M over the Y dimension. Each PE gets dN and dM many columns and rows to process, where dN = N / kernel_y_dim and dM = M / kernel_y_dim.
#  H is split across the X dimension, so each PE gets dH = H / kernel_x_dim many hidden dimension elements to reduce over.
#
#  We broadcast each column vector of size (dH) of dimension N across the Y dimension. This is done wavelet by wavelet. We use a SAXPY-like approach where each 
#  PE computes a partial reduction across the H dimension for its chunk of the M and N dimensions, and then we do a final reduction across the X dimension to 
#  produce the final output. This is repeated N times. Each PE computes a partial reduction of size {1, dM}. All PEs will reduce alon the X dimension to 
#  produce a single output column vector of size (dM) for each of the N columns, which are then concatenated together to produce the final output of size (M, N).
# 
# This creates the need for the following modules:
# 1) A module to broadcast the column vectors of B across the Y dimension, upon receival, we want to trigger a @fmach operation (SAXPY) for local accumulation.
# 2) A module to reduce the partial sums across the X dimension, which can be triggered after the SAXPY is done for each column vector. We always want to 
#    reduce into the next local memory vector location, but after dN many columns, we want to reduce to the next local PE until we got N many reductions.
assert M % kernel_y_dim == 0, "M must be divisible by kernel_y_dim"
assert H % kernel_x_dim == 0, "H must be divisible by kernel_x_dim"
assert N % kernel_y_dim == 0, "N must be divisible by kernel_y_dim"
assert N % kernel_x_dim == 0, "N must be divisible by kernel_x_dim"

# Construct A, B, C
A = np.random.rand(M*H).astype(np.float32).reshape(M,H)
B = np.random.rand(H*N).astype(np.float32).reshape(H,N)

# Calculate expected C
C_expected = np.matmul(A, B)

# Per-PE dimensions
dN_y = N // kernel_y_dim  # B columns per Y-PE
dN = N // kernel_x_dim    # C columns per X-PE
dH = H // kernel_x_dim
dM = M // kernel_y_dim

# Construct a runner using SdkRuntime (simfab traces enabled)
runner = SdkRuntime(args.name, cmaddr=args.cmaddr, suppress_simfab_trace=True)

# Get symbol for A on device
A_symbol = runner.get_id('A')
B_symbol = runner.get_id('B')
C_symbol = runner.get_id('C')

# Load and run the program
runner.load()
runner.run()


# Copy chunks of A, B into all PEs
# A is (M, H), split: M over Y-dim, H over X-dim
# Each PE gets a (dM, dH) chunk stored row-major
A_chunked = A.reshape(kernel_y_dim, dM, kernel_x_dim, dH).transpose(0, 2, 1, 3).transpose(0, 1, 3, 2)
A_prepared = A_chunked.ravel()
print(f"\n--- Local A values per PE (each PE has a {dM}x{dH} chunk) ---")
print(f"Global A [{A.shape}]:\n", A, )
for py in range(kernel_y_dim):
  for px in range(kernel_x_dim):
    chunk = A_chunked[py, px]
    hex_vals = np.vectorize(lambda f: format(np.float32(f).view(np.uint32), '08x'))(chunk)
    print(f"PE({px},{py}) A:\n{chunk}")
    print(f"PE({px},{py}) A (hex):\n{hex_vals}")
runner.memcpy_h2d(A_symbol, A_prepared, 0, 0, kernel_x_dim, kernel_y_dim, dM*dH,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# B is (H, N), split: H over X-dim, N over Y-dim
# Each PE gets a (dH, dN_y) chunk stored row-major
B_chunked = B.reshape(kernel_x_dim, dH, kernel_y_dim, dN_y).transpose(2, 0, 1, 3).transpose(0, 1, 3, 2)
B_prepared = B_chunked.ravel()
print(f"\nGlobal B [{B.shape}]:\n", B)
print(f"\n--- Local B values per PE (each PE has a {dH}x{dN_y} chunk) ---")
for py in range(kernel_y_dim):
  for px in range(kernel_x_dim):
    chunk = B_chunked[py, px]
    hex_vals = np.vectorize(lambda f: format(np.float32(f).view(np.uint32), '08x'))(chunk)
    print(f"PE({px},{py}) B:\n{chunk}")
    print(f"PE({px},{py}) B (hex):\n{hex_vals}")
runner.memcpy_h2d(B_symbol, B_prepared, 0, 0, kernel_x_dim, kernel_y_dim, dH*dN_y,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)


print("Reference values C per PE:")
print(f"Global C [{C_expected.shape}]:\n", C_expected)
C_expected_chunked = C_expected.reshape(kernel_y_dim, dM, kernel_x_dim, dN).transpose(0, 2, 1, 3)
for py in range(kernel_y_dim):
  for px in range(kernel_x_dim):
    chunk = C_expected_chunked[py, px]
    hex_vals = np.vectorize(lambda f: format(np.float32(f).view(np.uint32), '08x'))(chunk)
    print(f"PE({px},{py}) C expected:\n{chunk}")
    print(f"PE({px},{py}) C expected (hex):\n{hex_vals}")

# Start the boardcast of B.
runner.launch('broadcast_pe', nonblock=False)

# Print the local A, B values for each PE
A_readback = np.zeros([kernel_x_dim * kernel_y_dim * dM * dH], dtype=np.float32)
runner.memcpy_d2h(A_readback, A_symbol, 0, 0, kernel_x_dim, kernel_y_dim, dM*dH,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)
A_readback = A_readback.reshape(kernel_y_dim, kernel_x_dim, dM, dH)

B_readback = np.zeros([kernel_x_dim * kernel_y_dim * dH * dN_y], dtype=np.float32)
runner.memcpy_d2h(B_readback, B_symbol, 0, 0, kernel_x_dim, kernel_y_dim, dH*dN_y,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)
B_readback = B_readback.reshape(kernel_y_dim, kernel_x_dim, dH, dN_y)


# After the compute we want to load C to verify the results.
# C is distributed across PEs via the interleaved reduction ring.
# Each PE has a (dM, dN) chunk where dN = N / kernel_x_dim.
# Columns are block-distributed across X-PEs (dN consecutive columns per PE).
C_result = np.zeros([kernel_x_dim * kernel_y_dim * dM * dN], dtype=np.float32)
runner.memcpy_d2h(C_result, C_symbol, 0, 0, kernel_x_dim, kernel_y_dim, dM*dN,
  streaming=False, order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT,
  nonblock=False)

# Reshape C_result from PE layout back to (M, N)
# C is stored as C^T (dN x dM) row-major on each PE, so transpose per-PE
C_result = C_result.reshape(kernel_y_dim, kernel_x_dim, dN, dM).transpose(0, 1, 3, 2)
C_expected_chunked = C_expected.reshape(kernel_y_dim, dM, kernel_x_dim, dN).transpose(0, 2, 1, 3)
print(f"\n--- Local C values per PE (each PE has a {dM}x{dN} chunk) ---")
for py in range(kernel_y_dim):
  for px in range(kernel_x_dim):
    chunk = C_result[py, px]
    chunk_ref = C_expected_chunked[py, px]

    print(f"PE({px},{py})====================\n C:\n{chunk}")
    print(f"C expected:\n{chunk_ref}")
  
C_result = C_result.transpose(0, 2, 1, 3).reshape(M, N)
print("Ref   C result after reshaping to (M, N):\n", C_expected)
print("Final C result after reshaping to (M, N):\n", C_result)

# Stop the program
runner.stop()

# Ensure that the result matches our expectation
# TODO: The reshape/comparison below assumes C is block-distributed along the diagonal.
# Adjust based on your actual algorithm output distribution.
np.testing.assert_allclose(C_result, C_expected, rtol=1e-06, atol=1e-03)
print("Program completed. Verify C_result distribution matches expected.")
