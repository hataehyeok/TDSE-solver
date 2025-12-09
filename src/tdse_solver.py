import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu

'''
Arguments:
    - k: wavevector
    - a: lattice constant
    - v0: potential strength
    - ngx: number of grid points
    - nbnds: number of bands
    - Ngmax: maximum number of nearest neighbors

Returns:
    - E: energy eigenvalues
    - phi: eigenvectors

Detail:
    - Gx: frequence domain with centered at 0
    - b: reciprocal lattice constant (Primitive reciprocal lattice vector)
    - k: Wavenumber
    - lambda_kG: kinetic energy
    - Hij: Hamiltonian matrix
'''
def epsilon_nk_cosx_pot(k=0, a=10, v0=1.0, ngx=10, nbnds=5, Ngmax=1):
    Gx = np.arange(ngx, dtype=int)
    Gx[ngx // 2 + 1:] -= ngx

    b = 2*np.pi / a
    Gx = Gx * b
    k *= b

    lambda_kG = 0.5 * (k + Gx)**2
    Hij = np.diag(lambda_kG)

    nn = range(ngx)
    for ii in range(1, Ngmax+1):
        Hij[nn[:-ii], nn[ii:]] = v0
        Hij[nn[:-(ngx-ii)], nn[ngx-ii:]] = v0
        Hij[nn[ii:], nn[:-ii]] = v0
        Hij[nn[(ngx-ii):], nn[:-(ngx-ii)]] = v0
    
    E, phi = np.linalg.eigh(Hij) 

    return E[:nbnds], phi[:,:nbnds].T


'''
Arguments:
    - k0: center of the wavepacket
    - sigma_k: standard deviation of the wavepacket
    - nk: number of grid points
    - Nsigma: number of standard deviations

Returns:
    - k: wavevector
    - fk: Gaussian envelope

Detail:
    - ktmp: wavevector range
    - k1: wavevector range with centered at k0
    - k2: wavevector range with centered at k0
    - fk: Gaussian envelope
'''
def get_bloch_wavepacket_gaussian_envelop(k0, sigma_k, nk=20, Nsigma=10):
    assert -0.5 <= k0 <= 0.5

    delta_k = Nsigma * sigma_k
    ktmp    = np.linspace(0, delta_k, nk, endpoint=True)
    k1      = np.r_[-ktmp[::-1][:-1], ktmp]

    k2 = k1 + k0
    k2[k2 >= 0.5] -= 1.0
    k2[k2 <=-0.5] += 1.0
    fk = np.sqrt(1 / np.sqrt(np.pi) / sigma_k) * np.exp(-k1**2/(2*sigma_k**2))

    return k2, fk


'''
Arguments:
    - x: spatial grid
    - kwp: wavevector of the wavepacket
    - fk: Gaussian envelope
    - n0: band index
    - v0: potential strength
    - a: lattice constant
    - nbnds: number of bands
    - ngx: number of grid points

Returns:
    - bloch_wp: Bloch wavefunction

Detail:
    - Gx: frequence domain with centered at 0
    - b: reciprocal lattice constant (Primitive reciprocal lattice vector)
    - k: Wavenumber
    - lambda_kG: kinetic energy
    - Hij: Hamiltonian matrix
'''
def construct_blochwp_cosx_pot(x, kwp, fk, n0=0, 
        v0=0.05, a=1, nbnds=5, ngx=20):
    Nx  = x.size
    b   = 2*np.pi / a

    Cng = []
    for k in kwp:
        E, C = epsilon_nk_cosx_pot(k, a=a, v0=v0, ngx=ngx, nbnds=nbnds, Ngmax=1)
        Cng.append(C)
    Cng = np.array(Cng)

    Gx = np.arange(ngx, dtype=int)
    Gx[ngx // 2 + 1:] -= ngx

    bloch_wp = np.zeros(Nx, dtype=complex)
    for ik, k in enumerate(kwp):
        for ig, g in enumerate(Gx):
            bloch_wp[:] += np.exp(1j*(k+g)*x*b) * fk[ik] * Cng[ik,n0,ig]
    bloch_wp /= np.sqrt(np.trapz(np.abs(bloch_wp)**2, x))

    return bloch_wp


'''
Arguments:
    - psi0: initial wavefunction
    - V: potential
    - x: spatial grid
    - dt: time step
    - N: number of time steps
    - print_norm: print the norm of the wavefunction

Returns:
    - PSI_t: wavefunction at each time step

Detail:
    - J: number of grid points
    - dx: grid spacing
    - V: potential
    - O: ones
    - T: kinetic energy
    - U2: evolution operator
    - LU: LU decomposition of U2
    - PSI_t: wavefunction at each time step
'''
def CrankNicolson(psi0, V, x, dt, N=100, print_norm=False):
    J  = x.size - 1
    dx = x[1] - x[0]

    V = spa.diags(V)
    O = np.ones(J+1)
    T = (-1 / 2 / dx**2) * spa.spdiags([O, -2*O, O], [-1, 0, 1], J+1, J+1)

    U2 = spa.eye(J+1) + (1j * 0.5 * dt) * (T + V)
    U1 = spa.eye(J+1) - (1j * 0.5 * dt) * (T + V)
    U2 = U2.tocsc()
    LU = splu(U2)

    PSI_t = np.zeros((J+1, N), dtype=complex)
    PSI_t[:, 0] = psi0

    for n in range(N-1):
        b            = U1.dot(PSI_t[:,n])
        PSI_t[:,n+1] = LU.solve(b)
        if print_norm:
            print(n, np.trapz(np.abs(PSI_t[:,n+1])**2, x))

    return PSI_t


'''
Arguments:
    - psi0: initial wavefunction
    - V: potential
    - x: spatial grid
    - dt: time step
    - N: number of time steps
    - print_norm: print the norm of the wavefunction

Returns:
    - PSI_t: wavefunction at each time step

Detail:
    - Nx: number of grid points
    - dx: grid spacing
    - k: wavevector
    - Evol_T: time evolution operator
    - Evol_V_half: potential evolution operator
    - PSI_t: wavefunction at each time step
'''
def SplitStepFourier(psi0, V, x, dt, N=100, print_norm=False):
    Nx = x.size
    dx = x[1] - x[0]
    
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    
    Evol_T = np.exp(-1j * (k**2 * 0.5) * dt)
    Evol_V_half = np.exp(-1j * V * 0.5 * dt)

    PSI_t = np.zeros((Nx, N), dtype=complex)
    PSI_t[:, 0] = psi0
    
    psi = psi0.astype(complex)

    for n in range(N-1):
        psi *= Evol_V_half
        
        psi_k = np.fft.fft(psi)
        psi_k *= Evol_T
        psi = np.fft.ifft(psi_k)
        
        psi *= Evol_V_half

        PSI_t[:, n+1] = psi

        if print_norm:
            print(n, np.trapz(np.abs(psi)**2, x))

    return PSI_t
