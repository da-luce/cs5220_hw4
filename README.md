# Homework: Distributed Matrix Multiplication on WSE-2

## Objective

Implement $C = A \cdot B$ on a 2D grid of Cerebras processing elements, where $A \in \mathbb{R}^{M \times H}$, $B \in \mathbb{R}^{H \times N}$, $C \in \mathbb{R}^{M \times N}$. All elements are fp32.

The PE grid has `kernel_x_dim` columns and `kernel_y_dim` rows.

## What You Must Implement

You have three files to modify:

### 1. `config.py` — Data Layout Booleans

Set six boolean flags that control how matrices are distributed across PEs and stored in memory. The host-side infrastructure (`prepare_h2d` / `reconstruct_d2h`) in `run.py` imports these flags and uses them to chunk and reassemble the matrices automatically.

- `*_GLOBAL_TRANSPOSE`: controls whether a matrix's row axis maps to Y-PEs (False) or X-PEs (True).
- `*_MEMORY_TRANSPOSE`: controls whether each PE stores its local block transposed in memory.

Think carefully about which matrix dimension must align with the reduction axis (X), and which memory layout gives contiguous access for column-vector operations.

### 2. `layout.csl` — Fabric Routing

Inside the `layout { }` block, implement:

- **Tile code assignment**: Use `@set_tile_code` to assign `pe.csl` to each PE with the appropriate parameters. Each PE needs its own `reduce_tx_color` and `reduce_rx_color` — use the three provided reduction colors.
- **Reduction ring routing**: Configure `@set_color_config` for each PE so that partial sums flow from the last X-PE toward the first, with flyover colors for middle PEs.
- **Broadcast routing**: Configure `@set_color_config` so that each PE in a column can broadcast its B data to all other PEs in the same column. Use color swap at the head PE and switches (`SWITCH_ADV`) to sequence the senders.

### 3. `pe.csl` — On-PE Computation

Implement the per-PE logic:

- **Broadcast receive**: A data task that fires on each incoming wavelet, accumulates partial products via SAXPY (`fmac`), and triggers reduction when a full B-column has been received.
- **Reduction state machine**: A ring reduction across the X-dimension. PEs rotate through three roles (initiator, partial-sum, final-accumulator) across successive columns.
- **Broadcast send**: Send the local B data to the fabric, then send a control wavelet (`SWITCH_ADV`) to hand off to the next PE.
- **Completion**: Call `sys_mod.unblock_cmd_stream()` when all reductions finish.

## Provided Infrastructure

- `config.py`: Six layout booleans imported by `run.py`. This is the only Python file you need to edit.
- `run.py`: Matrix generation, `prepare_h2d`/`reconstruct_d2h` utilities, simulation launch, readback, and verification. **Do not modify** — this file will be reset by the autograder. You may temporarily edit it for debugging, but your changes will not be used in the submission.
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

# Check performance
./test_perf.sh
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
  (part of B down Y)        (accumulate A * b_col)       (sum partials along X)

       PE(x,0)                    All PEs                     PE row
       | b[h]                  red_in += A_col * b[h]     +---+---+---+
       v                      (repeated dH times)         |p0 |p1 |p2 | --> sum --> C_part
  +---------+                                             +---+---+---+
  | PE(x,1) |
  |   ...   |
  | PE(x,Y) |
  +---------+
```

### Phase 1 -- Column Broadcast

Each part of $B$ (a vector (col or row)) is broadcast from a single PE, (for all PEs over the total time of execution), in a column to all other PEs in that column.
The broadcast is split into a sending-up route and a common broadcast-down route. Because two directions cannot share the same color, the wavelets must change color at the head PE. PEs take turns sending: the top PE (y=0) sends first, then y=1, and so on, controlled by a fabric switch sentinel that advances after each PE finishes transmitting.

```
    Column x                     Routing

    PE(x,0)  <-- color swap -->  broadcast_rx_color sent NORTH is
       |         at head PE      reflected back SOUTH as broadcast_tx_color
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

$$r \mathrel{+}= A_j \cdot b_h$$

where $r$ is the local accumulation buffer `red_in`.

using the hardware `fmac` (fused multiply-accumulate) operation. After the PE accumulated enough data, the partial sum should be reduced along the row.

### Phase 3 -- Row Reduction (Ring)

The partial vectors `red_in` of length $dM$ are reduced (summed) across the X-dimension using a **ring**. The ring physically sends data EAST and returns WEST via flyover, forming a cycle through all PEs.

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

## Execution Flow

```
  Host                          Device (all PEs)
  ----                          ----------------
  memcpy A, B to PEs
          |
          v
  launch broadcast_pe()  --->  initialize_reduce_states pased on tile position
                                broadcast B
                                  |
                                  +-- recv wavelet --> SAXPY
                                  +-- after enough wavelets --> reduce()
                                  +-- after all N columns --> terminate_kernel()
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

## Debugging

### Simprint (`<simprint>`)

The `<simprint>` library lets you print values directly to the simulator log (`sim.log`) during execution. Simprint output appears immediately when a newline is encountered — making it invaluable for **debugging stalling programs**.

A helper function `simprint_pe_coords()` is provided in `pe.csl`. Call it before any print to identify the PE:

```csl
simprint_pe_coords();
prt.print_string("hello from this PE\n");

simprint_pe_coords();
prt.fmt_no_nl("current_row={d}\n", .{@as(u16, current_row)});

simprint_pe_coords();
prt.fmt_no_nl("reduce_count={d}, C[0]={f}\n", .{reduce_count, C[0]});
```

**Important:** Output is flushed to `sim.log` only when `"\n"` is encountered. Always end your format strings with `\n`.

For more details, study the `topic-13-simprint` tutorial.

### Reading `sim.log`

Each line in `sim.log` is prefixed with the cycle number and the absolute fabric coordinates of the PE:

```
@968  P4.1: Loc(0, 0) P4.1: sender beginning main_fn
@1156 P5.1: Loc(1, 0) P5.1: recv_task: in_data = 0, global = 0
@1888 P6.2: Loc(2, 1) P6.2: recv_task: in_data = 4, global = 20
```

- `@<cycle>` — simulator cycle number when the print fired.
- `P<abs_x>.<abs_y>` — absolute fabric coordinates (includes memcpy-infrastructure offsets).
- `Loc(<x>, <y>)` — logical PE coordinates within your kernel grid (printed by `simprint_pe_coords()`).

The absolute coordinates differ from the logical ones by the fabric offsets (typically `off_x=4, off_y=1`), so `Loc(0,0)` corresponds to `P4.1`.

### Extracting Per-PE Logs

Use `extract_logs.sh` to split `sim.log` into per-PE files:

```bash
./extract_logs.sh <sizeX> <sizeY>
```

This creates `sim_sprnt<abs_x>_<abs_y>.log` files containing only the simprint output for each PE.

### Debug Mode (Fabric Traces)

Run with `--debug` to enable fabric landing/router traces in `sim.log`:

```bash
./run.sh 2 2 4 4 4 --debug
```

This sets `SIMFABRIC_DEBUG=landing,router`, which logs every wavelet landing and routing decision. Use `--debug-instr` for full instruction traces (very verbose). The fabric trace lines show wavelet movement across the fabric and are useful for verifying that your routing configuration is correct.

## File Structure

| File | You Edit? | Description |
|------|-----------|-------------|
| `pe.csl` | **Yes** | Per-PE kernel written in CSL. This is where you implement the core computation. |
| `layout.csl` | **Yes** | Fabric topology and routing configuration. This is where you assign tile code to each PE via `@set_tile_code`, etc.. |
| `config.py` | **Yes** | Student configuration file. Contains six boolean flags (`A_GLOBAL_TRANSPOSE`, `A_MEMORY_TRANSPOSE`, `B_GLOBAL_TRANSPOSE`, `B_MEMORY_TRANSPOSE`, `C_GLOBAL_TRANSPOSE`, `C_MEMORY_TRANSPOSE`) that control how each matrix is chunked across the 2D PE grid and whether each PE's local block is stored transposed in memory. Imported by `run.py`. |
| `run.py` | No | Host-side Python driver executed via `cs_python`. Generates random matrices $A$ and $B$, distributes them across the PE grid using `prepare_h2d()`, launches the device kernel, reads back $C$ using `reconstruct_d2h()`, and verifies correctness against NumPy's `A @ B`. Imports layout booleans from `config.py`. Also reads back TSC timer buffers and prints min/max/mean cycle counts. **This file will be reset by the autograder** — you may temporarily edit it for debugging, but your changes will not be used in the submission. |
| `run.sh` | No | Compile-and-run wrapper. Invokes `cslc --arch=wse2` to compile `layout.csl` with the specified fabric dimensions and kernel parameters, then runs `cs_python run.py`. Accepts five positional arguments (`sizeX sizeY M H N`) and optional `--debug` (enables `SIMFABRIC_DEBUG=landing,router` for fabric traces) or `--debug-instr` (adds instruction-level traces). |
| `test_configs.sh` | No | Automated test sweep. Runs `run.sh` over ~30 configurations covering square grids, asymmetric grids, non-power-of-2 dimensions, large per-PE workloads, and edge cases. Reports PASS/FAIL for each. Use this to validate your implementation across a broad range of inputs. |
| `test_perf.sh` | No | Performance benchmark. Runs a single large configuration (8×8 grid, 128×256×64 matrices), then parses TSC timer statistics (min/max/mean cycles per PE) and simulator-reported total/idle/runtime cycle counts from `sim.log`. |
| `clean.sh` | No | Removes all build artifacts and logs: the `out/` compilation directory, `sim*.log` files, `wio_*` trace files, `simfab_traces/`, and JSON config files. Run before a fresh build. |
| `extract_logs.sh` | No | Post-simulation log splitter. Takes `sizeX` and `sizeY` as arguments, then greps `sim.log` to produce per-PE log files (`sim_sprnt<abs_x>_<abs_y>.log`) containing only that PE's simprint output. Useful for isolating a single PE's debug trace from a multi-PE run. |

