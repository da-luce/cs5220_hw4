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

# --- Matrix A (M x H) ---
A_GLOBAL_TRANSPOSE = ???  # TODO
A_MEMORY_TRANSPOSE = ???  # TODO

# --- Matrix B (H x N) ---
B_GLOBAL_TRANSPOSE = ???  # TODO
B_MEMORY_TRANSPOSE = ???  # TODO

# --- Matrix C (M x N) ---
C_GLOBAL_TRANSPOSE = ???  # TODO
C_MEMORY_TRANSPOSE = ???  # TODO
