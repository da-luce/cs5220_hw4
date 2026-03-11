# Homework: Distributed Matrix Multiplication on WSE-2

## Objective

Implement $C = A \cdot B$ on a 2D grid of Cerebras processing elements, where $A \in \mathbb{R}^{M \times H}$, $B \in \mathbb{R}^{H \times N}$, $C \in \mathbb{R}^{M \times N}$.

The PE grid has `kernel_x_dim` columns and `kernel_y_dim` rows.

## What You Must Implement

You have three files to modify:

### 1. `run.py` — Data Layout Booleans

Set six boolean flags that control how matrices are distributed across PEs and stored in memory. The host-side infrastructure (`prepare_h2d` / `reconstruct_d2h`) is provided and uses these flags to chunk and reassemble the matrices automatically.

- `*_GLOBAL_TRANSPOSE`: controls whether a matrix's row axis maps to Y-PEs (False) or X-PEs (True).
- `*_MEMORY_TRANSPOSE`: controls whether each PE stores its local block transposed in memory.

Think carefully about which matrix dimension must align with the reduction axis (X), and which memory layout gives contiguous access for column-vector operations.

### 2. `layout.csl` — Fabric Routing

Inside the `layout { }` block, implement:

- **Tile code assignment**: Use `@set_tile_code` to assign `pe.csl` to each PE with the appropriate parameters. Each PE needs its own `reduce_tx_color` and `reduce_rx_color` — use the three provided reduction colors to build an interleaved ring.
- **Reduction ring routing**: Configure `@set_color_config` for each PE so that partial sums flow from the last X-PE toward the first, with flyover colors for middle PEs.
- **Broadcast routing**: Configure `@set_color_config` so that each PE in a column can broadcast its B data to all other PEs in the same column. Use color swap at the head PE and switches (`SWITCH_ADV`) to sequence the senders.

### 3. `pe.csl` — On-PE Computation

Implement the per-PE logic:

- **Broadcast receive**: A data task that fires on each incoming wavelet, accumulates partial products via SAXPY (`fmac`), and triggers reduction when a full B-column has been received.
- **Reduction state machine**: An interleaved ring reduction across the X-dimension. PEs rotate through three roles (initiator, partial-sum, final-accumulator) across successive columns.
- **Broadcast send**: Send the local B data to the fabric, then send a control wavelet (`SWITCH_ADV`) to hand off to the next PE.
- **Completion**: Call `sys_mod.unblock_cmd_stream()` when all reductions finish.

## Provided Infrastructure

- `run.py`: Matrix generation, `prepare_h2d`/`reconstruct_d2h` utilities, simulation launch, readback, and verification. You only set the six `???` booleans.
- `layout.csl`: Color map, memcpy boilerplate, dimension asserts, `common_params` struct.
- `pe.csl`: Parameter declarations, module imports, queue declarations, storage arrays (`A`, `B`, `C`), export symbols.
- `run.sh`, `test_configs.sh`, `clean.sh`, `extract_logs.sh`: Build and test scripts.

## Running

```bash
# Single configuration
./run.sh <sizeX> <sizeY> <M> <H> <N>

# All test configurations
./test_configs.sh

# Debug mode (fabric traces)
./run.sh 2 2 4 4 4 --debug
```

## Constraints

- `kernel_x_dim >= 2`, `kernel_y_dim >= 2`
- `M % kernel_y_dim == 0`, `H % kernel_x_dim == 0`
- `N % kernel_x_dim == 0`, `N % kernel_y_dim == 0`

## Recommended Tutorials

Study these examples before starting:

- `topic-06-switches` — fabric switches and SWITCH_ADV
- `topic-14-color-swap` — color swap routing
- `gemv-06-routes-1` through `gemv-08-routes-3` — fabric routing patterns
- `topic-11-collectives` — broadcast and reduce patterns
- `sdklayout-02-routing` — `@set_color_config` API

```
  Phase 1: Broadcast          Phase 2: Local SAXPY         Phase 3: Row Reduce
  (column of B down Y)        (accumulate A * b_col)       (sum partials along X)

       PE(x,0)                    All PEs                     PE row
       | b[h]                  red_in += A_col * b[h]     +---+---+---+
       v                      (repeated dH times)         |p0 |p1 |p2 | --> sum --> C_col
  +---------+                                             +---+---+---+
  | PE(x,1) |
  |   ...   |
  | PE(x,Y) |
  +---------+
```

### Phase 1 -- Column Broadcast

Each column of $B$ (a vector of length $dH$) is broadcast from every PE in a column to all other PEs in that column, using a **color-swap** routing pattern along the Y-axis. PEs take turns sending: the top PE (y=0) sends first, then y=1, and so on, controlled by a fabric switch sentinel that advances after each PE finishes transmitting.

```
    Column x                     Routing

    PE(x,0)  <-- color swap -->  tx_color sent NORTH is reflected
       |         at head PE      back SOUTH as rx_color
       v
    PE(x,1)  -- forward SOUTH
       |
       v
    PE(x,2)  -- terminal (no further forwarding)

  After PE(x,0) finishes, a SWITCH_ADV control wavelet
  shifts the source to PE(x,1), then PE(x,2), etc.
```

Each wavelet carries one `f32` element. Over the full broadcast of one B-column, $dH \times dN_y$ wavelets are sent per PE. All `kernel_y_dim` PEs in the column transmit sequentially, so every PE receives the B-column data from all other PEs in the column.

### Phase 2 -- Local SAXPY

Upon receiving each broadcast wavelet $b_h$ (a scalar from $B$), the PE executes:

$$r[0:dM] \mathrel{+}= A[:,h] \cdot b_h$$

where $r$ is the local accumulation buffer `red_in`.

using the hardware `fmac` (fused multiply-accumulate) operation. After $dH$ wavelets (one full column of $B$'s local block), the accumulation buffer holds the local partial product $A \cdot b$, a vector of length $dM$.

### Phase 3 -- Row Reduction (Ring)

The partial vectors `red_in` of length $dM$ are reduced (summed) across the X-dimension using a **3-color ring**. The ring physically sends data EAST on two alternating colors (A, B) and returns WEST on a third color (C) via flyover, forming a cycle through all PEs.

```
  Ring across X-dimension (kernel_x_dim = 4 example):

  State rotation for successive reduce() calls:

  Call #    PE0      PE1      PE2      PE3      Chain (ISUM -> ... -> FSUM)
  -----   ------   ------   ------   ------    ---------------------------
    0      FSUM     ISUM     PSUM     PSUM     PE1 -> PE2 -> PE3 -> PE0
    1      PSUM     FSUM     ISUM     PSUM     PE2 -> PE3 -> PE0 -> PE1
    2      PSUM     PSUM     FSUM     ISUM     PE3 -> PE0 -> PE1 -> PE2
    3      ISUM     PSUM     PSUM     FSUM     PE0 -> PE1 -> PE2 -> PE3

  ISUM: Initiator -- send local red_in to fabric (no receive)
  PSUM: Partial   -- receive from fabric, add local red_in, forward
  FSUM: Final     -- receive from fabric, add local red_in, write to C
```

Each PE maintains a state machine that cycles through ISUM, PSUM, and FSUM states. The ISUM PE injects its partial sum; each subsequent PSUM PE receives, adds its own partial sum, and forwards; the FSUM PE writes the fully accumulated result to its local $C$. After `kernel_x_dim` reductions, each PE has written one column to its local $C$ block via the rotating FSUM assignment.

Routing uses three fabric colors (A, B, C) to avoid conflicts on shared wires between adjacent PEs:

```
  Physical ring topology:

  PE0          PE1          PE2          PE3
  tx:A rx:C    tx:B rx:A    tx:A rx:B    tx:C rx:A
      ---A--->     ---B--->     ---A--->
      <---C (flyover through middle PEs)---

  Upper leg (EAST):  PE0 --A--> PE1 --B--> PE2 --A--> PE3
  Return leg (WEST): PE3 --C--> (flyover PE2, PE1) --> PE0
```

## Execution Flow

```
  Host                          Device (all PEs)
  ----                          ----------------
  memcpy A, B to PEs
          |
          v
  launch broadcast_pe()  --->  initialize_reduce()
                                broadcast B column-by-column
                                  |
                                  +-- recv wavelet --> SAXPY
                                  +-- after dH wavelets --> reduce()
                                  +-- after all N columns --> unblock
          |
          v
  memcpy C from PEs
  verify C == A * B
```

Total reductions per PE: `kernel_y_dim` $\times$ `dN_y` $= N$, one per global column of $C$.

## Constraints

- `kernel_x_dim >= 2`, `kernel_y_dim >= 2`
- $M$ mod `kernel_y_dim` $= 0$
- $H$ mod `kernel_x_dim` $= 0$
- $N$ mod `kernel_x_dim` $= 0$
- $N$ mod `kernel_y_dim` $= 0$

## File Structure

| File              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `layout.csl`      | Fabric topology: color routing, switch config, PE params |
| `pe.csl`          | PE kernel: broadcast send/recv, SAXPY, reduce state machine |
| `run.py`          | Host driver: data distribution, launch, verification |
| `run.sh`          | Compile and execute for a single configuration       |
| `test_configs.sh` | Sweep over parameter configurations                  |
