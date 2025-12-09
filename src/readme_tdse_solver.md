# Document about `tdse_solver.py`

## Used Variables

### System & Potential

- `a`: lattice constant
- `v0`: potential strength
- `b`: reciprocal lattice vector, $b = \frac{2\pi}{a}$
- `Vx`: total potential, $V_x = 2v_0 \cos(bx) -qEx$

### Force & Time

- `qE`: external force
- `dt`: time step (time interval of simulation)
- `Tbloch`: bloch period, $T_B = \frac{2\pi}{qE \cdot a}$, spse $\hbar=1$
- `NSW`: number of steps, total number of simulation steps (we set it as `6 * Tbloch`)

### Numerical & Basis

- `ngx`:
  - number of G-vectors
  - number of basis that for plane wave expansion
  - the higher value, the more precise energy calculation becomes
- `nbnds`: number of bands to calculate
- `Gx`: 
  - integer index array of reciprocal lattice vector
  - it is used to construct the plane wave basis $e^{i(k+G)x}$

### Spatial Grid

- `Nc`: number of cells, contained in simulation domain
- `Nx`: grid points, discretize the space into system of gird points
- `L`: total length, $L = N_c \times a$
- `x`: coordinate space

### Wavefunction & Band Structure**

- `k0_center`: center crystal momentum of the initial Gaussian wavepacket
- `sigma_k`: width (spread) of the wavepacket in k-space
- `kwp`: array of k-points sampled for the wavepacket construction
- `fk`: Gaussian envelope coefficients (weights) corresponding to `kwp`
- `bloch_wp`: initial wavefunction at $t=0$ constructed by superposition of bloch states
- `k_1d_bz`: array of k-values discretizing the 1st Brillouin Zone ,$-\pi/a \sim \pi/a$
- `Enk`: energy eigenvalues for each k (band structure)
- `phi_nk`: bloch eigenstates corresponding to each k
- `PSI0`: computed time-dependent wavefunction $\psi(x,t)$

## Algorithm of `run_bloch_simulation`

This is the mathematical formulation of bloch simulation

### Hamiltonian Setup

- $\hat{H}$ corresponds to `Vx` in the code
    $$\hat{H} = -\frac{1}{2}\frac{\partial^2}{\partial x^2} + 2v_0 \cos\left(\frac{2\pi}{a}x\right) - qEx$$

### Band Structure Calculation

- It corresponds to for `k in k_1d_bz:` in the code
- Solve the TISE through **Plane Wave Explasion** for band structure 
    $$\psi_{n,k}(x) = \frac{1}{\sqrt{L}} \sum_{G} C_{n,G}(k) e^{i(k+G)x}$$
- Compute the 
    $$\sum_{G'} \left[ \frac{1}{2}(k+G)^2 \delta_{G,G'} + V_{G-G'} \right] C_{n,G'}(k) = E_n(k) C_{n,G}(k)$$

### Initial Wavepacket

- `block_wp` is constructed as a superposition of Bloch states weighed by a Gaussian envelop $f(k)$
    $$\Psi(x, t=0) = \int_{BZ} f(k) \psi_{n=0, k}(x) \, dk$$
    $$f(k) \propto \exp\left(-\frac{(k-k_0)^2}{2\sigma_k^2}\right)$$

### Time Propagation

- Solve the TDSE
    $$i\frac{\partial}{\partial t}\Psi(x,t) = \hat{H}\Psi(x,t)$$
- **Split-Step Fourier Method (SSFM):**
    $$\Psi(t+\Delta t) \approx e^{-i\hat{V}\Delta t/2} \mathcal{F}^{-1} \left[ e^{-i\hat{T}\Delta t} \mathcal{F} \left[ e^{-i\hat{V}\Delta t/2} \Psi(t) \right] \right]$$
- **Crank-Nicolson Method (CN):**
    $$\left(1 + \frac{i\hat{H}\Delta t}{2}\right) \Psi(t+\Delta t) = \left(1 - \frac{i\hat{H}\Delta t}{2}\right) \Psi(t)$$

```python
def run_bloch_simulation(a, v0, qE, dt, ngx, nbnds, k0_center, sigma_k, 
                         Nc, Nx, x_range, n_periods, solver, filename, frame_skip, fps=25):
    b = 2 * np.pi / a
    Tbloch = 2 * np.pi / (qE * a)
    L = Nc * a
    
    x = np.linspace(x_range[0] * L, x_range[1] * L, Nx)
    Vx = 2 * v0 * np.cos(b * x) - qE * x
    
    kwp, fk = get_bloch_wavepacket_gaussian_envelop(k0_center, sigma_k=sigma_k, nk=100, Nsigma=10)
    bloch_wp = construct_blochwp_cosx_pot(x, kwp, fk, a=a, v0=v0, nbnds=nbnds, ngx=ngx)
    
    k_1d_bz = np.linspace(-0.5, 0.5, 100)
    Gx = np.arange(ngx, dtype=int)
    Gx[ngx // 2 + 1:] -= ngx
    
    Enk, phi_nk = [], []
    for k in k_1d_bz:
        e, c = epsilon_nk_cosx_pot(k, a=a, v0=v0, ngx=ngx, nbnds=nbnds, Ngmax=1)
        Enk.append(e)
        phi_nk.append(np.sum(c[0][:, None] * np.exp(1j * b * (Gx + k)[:, None] * x[None, :]), axis=0))
    
    Enk, phi_nk = np.asarray(Enk), np.asarray(phi_nk)
    
    NSW = n_periods * int(Tbloch / dt)
    PSI0 = solver(bloch_wp, Vx, x, dt, NSW, False)
    
    create_bloch_animation(
        PSI0, Enk, phi_nk, Vx, x, k_1d_bz,
        params={'a': a, 'dt': dt, 'Tbloch': Tbloch, 'n0': 0, 'nbnds': nbnds},
        filename=filename,
        frame_skip=frame_skip,
        fps=fps
    )
```
