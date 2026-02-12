# TSA-Benchmark


> **"Bro trust me, my neural data lives in a 7-dimensional manifold"**

<p align="center">
<img src="https://usagif.com/wp-content/uploads/gify/are-you-sure-about-that-21-usagif.gif" width="25%" height="25%"/>
</p>


Here's a synthetic benchmark for **Time Series Analysis** methods, built to challenge them in recovering hidden manifold when we **actually** have a ground truth.

The core idea: generate smooth, low-dimensional dynamics on a known manifold, embed them into a high-dimensional observation space, then shuffle up the spatial structure, and finally spice up with noise.


## The generative process

| $S^D$ sphere   | Particle Dynamics  | Random linear map     |  Shuffle rows |
|---|---|---|---|
| latent manfiold  | $T$-steps trajectories  | observable space |  shuffled (destroy grouping) |
| P particles  in  $S^D$  | [P, D+1, T]  |  [P, N, T]   | [P*N, T]  | 



<p align="center">
  <img src="https://github.com/0xbadc0ffe/TSA-Benchamrk/blob/main/imgs/particles.png" />
</p>

### CHALLENGE:
1. Can you recover the (P(D+1)) dynamic components?
2. Can you recover the number of particles?
3. Can you revocer the manifold dimension?
 


### Step 1 — Smooth dynamics on $S^D$

Each of the **P** particles lives on a **D-dimensional sphere** $S^D$ (embedded in $\mathbb R^{D+1}$).
Motion is governed by **Brownian acceleration with damping**:

| Quantity | Update rule |
|---|---|
| Acceleration |$a_t \sim N(0, \sigma_{acc} ^2 I)$, projected to tangent space $T_{p_t} S^D$ |
| Velocity | $v_{t+1} = (1 - \gamma dt) (v_t + a_t dt)$, re-projected to tangent space |
| Position | $p_{t+1} = Exp_{p_t}(v_{t+1} dt)$ via the geodesic exponential map |

This produces **$C^1$-smooth trajectories** that remain exactly on the sphere (up to floating-point normalization). The damping term `gamma` prevents velocity from growing unboundedly.

**Initial conditions** per particle:
- Position: uniform random on $S^D$ (via normalized Gaussian)
- Velocity: random tangent vector at the initial position

**Optionally** you can define a spiking probability on the accelaration and velocity.

### Step 2 — Random linear embedding

A fixed random matrix **A** of shape `(N, D+1)` with entries drawn from $\mathcal{N}(0, 1/\sqrt{(D+1)})$ maps each time-step from the sphere's ambient space $R^{D+1}$ into a higher-dimensional observation space $R^N$.

This produces `P` trajectories in R^N. 

### Step 3 — Shuffle

The P*N rows (N observation channels per particle, P particles) are **randomly permuted**, destroying any spatial grouping. Only the **temporal ordering** (columns) is preserved.

The final output is a matrix of shape **(P\*N, T)** — the benchmark input.


### Noise

Gaussian noise can be added at two levels:
- **Trajectory noise** (`traj_noise_std`): added to sphere coordinates before projection
- **Observation noise** (`noise_std`): added to the observed signals after projection
  
[**TODO**: better noise models]


## What makes this challenging

The observed data has hidden structure at multiple levels:

| Property | Ground truth | What a method must recover |
|---|---|---|
| **Dynamic components** | P(D+1) independent signals | Separate the mixed components from P*N shuffled channels |
| **Number of particles** | P | Identify which channels belong to the same underlying source |
| **Manifold dimension** | D | Detect that the data lives on a D-dimensional manifold (not D+1 or N) |
| **Temporal smoothness** | $C^1$ trajectories | Exploit smooth dynamics vs noise |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `P` | 10 | Number of particles |
| `T` | 500 | Number of time steps |
| `D` | 2 | Sphere dimension (S^D in R^{D+1}) |
| `N` | 50 | Observation space dimension |
| `dt` | 0.05 | Integration time step |
| `sigma_acc` | 1.0 | Brownian acceleration noise std |
| `damping` | 0.5 | Velocity damping coefficient |
| `traj_noise_std` | 0.0 | Noise on sphere coordinates (before projection) |
| `noise_std` | 0.0 | Observation noise (after projection) |
| `seed` | 42 | Random seed for reproducibility |

## Quick start

```bash
pip install numpy matplotlib scikit-learn
```

```python
from tsa_bench import generate_sphere_benchmark

data, meta = generate_sphere_benchmark(P=10, T=500, D=2, N=50)

# data:                (PN, T) = (500, 500)  — the benchmark input
# meta['shuffle_idx']: the row permutation (ground truth)
# meta['trajectories']: (P, D+1, T) true sphere coordinates
# meta['A']:           (N, D+1) the random projection matrix
# meta['observed']:    (P, N, T) observations before shuffling
```

See [test.ipynb](test.ipynb) for a full example with visualizations and a PCA baseline.

## Difficulty scaling

| Easier | Harder |
|---|---|
| Small N (close to D+1) | Large N (high overcompleteness) |
| No noise | High `noise_std` and `traj_noise_std` |
| Few particles (small P) | Many particles (large P) |
| Low D | High D (curse of dimensionality) |
| Long T | Short T (less temporal signal) |

## License

MIT
