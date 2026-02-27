# Distributed Matrix Multiplication on WSE-2

## Problem

Compute $C = A \cdot B$ where $A \in \mathbb{R}^{M \times H}$, $B \in \mathbb{R}^{H \times N}$, $C \in \mathbb{R}^{M \times N}$.

The computation is mapped onto a 2D grid of `kernel_x_dim` x `kernel_y_dim` processing elements (PEs).

## Data Distribution

The hidden dimension $H$ is partitioned across the X-axis. Each PE holds a block of $A$ and $B$ and accumulates a partial product that must be reduced along X to form the final result.

```
                         Global A (M x H)                    Global B (H x N)
                    +--------+--------+----+            +--------+--------+----+
                    | dH col | dH col |    |            | dN_y   | dN_y   |    |
                    |   x=0  |   x=1  |... |            |  y=0   |  y=1   |... |
              +-----+--------+--------+----+      +-----+--------+--------+----+
              | dM  |A(0,0)  |A(0,1)  |    |      | dH  |B(0,0)  |B(0,1)  |    |
         y=0  |rows |dM x dH |dM x dH |    |  x=0 |rows |dH x dN |dH x dN |    |
              +-----+--------+--------+----+      +-----+--------+--------+----+
              | dM  |A(1,0)  |A(1,1)  |    |      | dH  |B(1,0)  |B(1,1)  |    |
         y=1  |rows |        |        |    |  x=1 |rows |        |        |    |
              +-----+--------+--------+----+      +-----+--------+--------+----+
              |     |  ...   |  ...   |    |      |     | ...    | ...    |    |
              +-----+--------+--------+----+      +-----+--------+--------+----+

  dM = M / kernel_y_dim       dH = H / kernel_x_dim
  dN = N / kernel_x_dim       dN_y = N / kernel_y_dim
```

Each PE at grid position $(x, y)$ stores:

| Matrix | Block        | Dimensions     | Index range                                     |
|--------|------------- |----------------|-------------------------------------------------|
| A      | A(y, x)      | dM x dH        | rows [y*dM : (y+1)*dM], cols [x*dH : (x+1)*dH] |
| B      | B(x, y)      | dH x dN_y      | rows [x*dH : (x+1)*dH], cols [y*dN_y : (y+1)*dN_y] |
| C      | C(y, x)      | dM x dN         | rows [y*dM : (y+1)*dM], cols [x*dN : (x+1)*dN] |

## Algorithm Overview

The algorithm proceeds in $N$ outer steps, one per global column of $C$. Each step computes one column of $C$ via a **column broadcast** of $B$ followed by a **row reduction** of partial sums.

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

$$\texttt{red\_in}[0\!:\!dM] \mathrel{+}= A[\,:\,,\, h] \cdot b_h$$

using the hardware `fmac` (fused multiply-accumulate) operation. After $dH$ wavelets (one full column of $B$'s local block), the accumulation buffer `red_in` holds the local partial product $A_{\text{local}} \cdot b_{\text{col}}$, a vector of length $dM$.

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

Total reductions per PE: $\texttt{kernel\_y\_dim} \times dN_y = N$, one per global column of $C$.

## Constraints

- `kernel_x_dim >= 2`, `kernel_y_dim >= 2`
- $M \bmod \texttt{kernel\_y\_dim} = 0$
- $H \bmod \texttt{kernel\_x\_dim} = 0$
- $N \bmod \texttt{kernel\_x\_dim} = 0$
- $N \bmod \texttt{kernel\_y\_dim} = 0$

## File Structure

| File              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `layout.csl`      | Fabric topology: color routing, switch config, PE params |
| `pe.csl`          | PE kernel: broadcast send/recv, SAXPY, reduce state machine |
| `run.py`          | Host driver: data distribution, launch, verification |
| `run.sh`          | Compile and execute for a single configuration       |
| `test_configs.sh` | Sweep over parameter configurations                  |
