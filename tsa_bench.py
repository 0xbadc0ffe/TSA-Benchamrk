import numpy as np
import matplotlib.pyplot as plt


def sample_sphere_uniform(rng, D, n_samples=1):
    """Sample n_samples points uniformly on S^D (in R^{D+1})."""
    x = rng.normal(size=(n_samples, D + 1))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def project_tangent(v, p):
    """Project vector v onto the tangent space of S^D at point p.
    
    v: (..., D+1), p: (..., D+1)
    """
    return v - np.sum(v * p, axis=-1, keepdims=True) * p


def generate_sphere_benchmark(P=10, T=500, D=2, N=50, dt=0.05,
                               sigma_acc=1.0, damping=0.5,
                               traj_noise_std=0.0, noise_std=0.0, seed=42):
    """
    Generate a time series benchmark: P particles on S^D mapped to R^N.
    
    Parameters
    ----------
    P : int
        Number of particles.
    T : int
        Number of time steps.
    D : int
        Sphere dimension (S^D lives in R^{D+1}).
    N : int
        Observation space dimension.
    dt : float
        Integration time step.
    sigma_acc : float
        Standard deviation of Brownian acceleration noise.
    damping : float
        Velocity damping coefficient (friction).
    traj_noise_std : float
        Standard deviation of noise added to trajectories (before projection).
    noise_std : float
        Observation noise standard deviation.
    seed : int or None
        Random seed.
    
    Returns
    -------
    data_shuffled : ndarray, shape (P*N, T)
        Shuffled observed time series.
    meta : dict with keys:
        'shuffle_idx': permutation applied to rows
        'trajectories': (P, D+1, T) true sphere coords
        'A': (N, D+1) random projection matrix
        'observed': (P, N, T) observed data before shuffling
    """
    rng = np.random.default_rng(seed)
    ambient_dim = D + 1
    
    trajectories = np.zeros((P, ambient_dim, T))
    
    for i in range(P):
        # Random initial position on S^D
        pos = sample_sphere_uniform(rng, D)[0]
        
        # Random initial velocity (tangent to sphere)
        vel = rng.normal(size=ambient_dim) * 0.5
        vel = project_tangent(vel, pos)
        
        for t in range(T):
            trajectories[i, :, t] = pos
            
            # Brownian acceleration in tangent space
            acc = rng.normal(0, sigma_acc, size=ambient_dim)
            acc = project_tangent(acc, pos)
            
            # Update velocity: acceleration + damping
            vel = vel + acc * dt
            vel = vel * (1 - damping * dt)
            vel = project_tangent(vel, pos)  # maintain tangency
            
            # Geodesic step via exponential map
            step = vel * dt
            step_norm = np.linalg.norm(step)
            if step_norm > 1e-12:
                pos = np.cos(step_norm) * pos + np.sin(step_norm) / step_norm * step
                pos /= np.linalg.norm(pos)  # numerical safety
            
            # Re-project velocity to tangent space at new position
            vel = project_tangent(vel, pos)

    # Add trajectory noise
    if traj_noise_std > 0:
        trajectories = trajectories + rng.normal(0, traj_noise_std, size=trajectories.shape)
    
    # Random linear map R^{D+1} -> R^N
    A = rng.normal(0, 1.0 / np.sqrt(ambient_dim), size=(N, ambient_dim))
    
    # Map trajectories to observation space
    observed = np.einsum('nd,pdt->pnt', A, trajectories)
    
    # Add observation noise
    if noise_std > 0:
        # observed = observed + rng.normal(0, noise_std, size=observed.shape)
        for k in range(observed.shape[0]):
            noisevec = rng.normal(0, noise_std, size=observed[k].shape[-1])
            observed[k] += noisevec
            # if k<observed.shape[0]-1:
            #     observed[k+1] += noisevec
    
    # Reshape to (P*N, T) and shuffle rows
    data = observed.reshape(P * N, T)
    shuffle_idx = rng.permutation(P * N)
    data_shuffled = data[shuffle_idx]
    
    meta = {
        'shuffle_idx': shuffle_idx,
        'trajectories': trajectories,
        'A': A,
        'observed': observed,
    }
    return data_shuffled, meta