# =============================================================================
# Student Configuration: Data Layout Booleans
# =============================================================================
#
# You are computing C = A * B where A ∈ R^{M×H}, B ∈ R^{H×N}, C ∈ R^{M×N}.
#
# The PE grid has kernel_x_dim columns and kernel_y_dim rows.
#
# Global layout controls how matrix blocks map to PEs:
#   False = "natural": matrix-rows along Y-PEs, matrix-cols along X-PEs
#   True  = "transposed": matrix-rows along X-PEs, matrix-cols along Y-PEs
#
# Memory layout controls how each local block is stored:
#   False = row-major in natural orientation
#   True  = store the transpose (row-major of the transposed block)
#
# Hint: think about which dimension needs to be contiguous in memory for
# efficient DSD access (broadcast columns of B, SAXPY with columns of A,
# write columns into C).
# =============================================================================

# Every memory transpose decision is made purely to ensure that
# Data Structure Descriptors (DSDs) are interacting with flat, contiguous
# arrays of data

# --- Matrix A (M x H) ---
# The Natural layout. The M dimension maps to the Y-axis (PE rows) and  the H
# dimension maps to the X-axis (PE cols).
A_GLOBAL_TRANSPOSE = False  

# Store column-major. In Phase 2 (Local SAXPY), you compute 
# r += A_j * b_h. To do this efficiently, the 1D DSD needs to read a full 
# contiguous column of A (A_j) from memory in one fast sweep.
A_MEMORY_TRANSPOSE = True

# --- Matrix B (H x N) ---
# GLOBAL: Transposed layout. This rotates B so its H dimension maps to the 
# X-axis, perfectly aligning with A's H dimension. This guarantees that 
# every PE in a specific column shares the same inner H index, which is 
# required for the Phase 1 column broadcast to work.
B_GLOBAL_TRANSPOSE = True

# MEMORY: Store row-major. Because we already globally transposed B, the host 
# sends it to the device as an N x H matrix. By keeping it row-major, those 
# H-length rows (which are the original columns of B!) are perfectly contiguous 
# in local memory. This allows the Phase 1 sender to broadcast the column 
# using a simple, fast 1D DSD.
B_MEMORY_TRANSPOSE = False

# --- Matrix C (M x N) ---
# GLOBAL: Natural layout. The final output dimensions M (Y-axis) and N (X-axis) 
# map to the grid normally, matching the physical layout of the Phase 3 
# reduction ring.
C_GLOBAL_TRANSPOSE = False

# MEMORY: Store column-major. In Phase 3, the reduction ring passes M-length 
# column vectors of partial sums. When a PE is in the FSUM state, it needs to 
# write that accumulated column vector directly to local memory. Storing C in 
# column-major order ensures the destination is contiguous.
C_MEMORY_TRANSPOSE = True