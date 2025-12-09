import numpy as np
import scipy.sparse as spa
from scipy.sparse.linalg import splu
from helper import create_bloch_animation


def epsilon_nk_cosx_pot(k=0, a=10, v0=1.0, ngx=10, nbnds=5, Ngmax=1):
    Gx = np.arange(ngx, dtype=int)
    Gx[ngx // 2 + 1:] -= ngx

    b = 2 * np.pi / a
    Gx = Gx * b
    k *= b

    lambda_kG = 0.5 * (k + Gx)**2
    Hij = np.diag(lambda_kG)

    nn = range(ngx)
    for ii in range(1, Ngmax + 1):
        Hij[nn[:-ii], nn[ii:]] = v0
        Hij[nn[:-(ngx-ii)], nn[ngx-ii:]] = v0
        Hij[nn[ii:], nn[:-ii]] = v0
        Hij[nn[(ngx-ii):], nn[:-(ngx-ii)]] = v0

    E, phi = np.linalg.eigh(Hij)
    return E[:nbnds], phi[:, :nbnds].T


def get_bloch_wavepacket_gaussian_envelop(k0, sigma_k, nk=20, Nsigma=10):
    assert -0.5 <= k0 <= 0.5

    delta_k = Nsigma * sigma_k
    ktmp = np.linspace(0, delta_k, nk, endpoint=True)
    k1 = np.r_[-ktmp[::-1][:-1], ktmp]

    k2 = k1 + k0
    k2[k2 >= 0.5] -= 1.0
    k2[k2 <= -0.5] += 1.0
    fk = np.sqrt(1 / np.sqrt(np.pi) / sigma_k) * np.exp(-k1**2 / (2 * sigma_k**2))

    return k2, fk


def construct_blochwp_cosx_pot(x, kwp, fk, n0=0, v0=0.05, a=1, nbnds=5, ngx=20):
    Nx = x.size
    b = 2 * np.pi / a

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
            bloch_wp += np.exp(1j * (k + g) * x * b) * fk[ik] * Cng[ik, n0, ig]
    bloch_wp /= np.sqrt(np.trapz(np.abs(bloch_wp)**2, x))

    return bloch_wp


def CrankNicolson(psi0, V, x, dt, N=100, print_norm=False):
    J = x.size - 1
    dx = x[1] - x[0]

    V = spa.diags(V)
    O = np.ones(J + 1)
    T = (-1 / 2 / dx**2) * spa.spdiags([O, -2*O, O], [-1, 0, 1], J + 1, J + 1)

    U1 = spa.eye(J + 1) - (1j * 0.5 * dt) * (T + V)
    U2 = spa.eye(J + 1) + (1j * 0.5 * dt) * (T + V)
    LU = splu(U2.tocsc())

    PSI_t = np.zeros((J + 1, N), dtype=complex)
    PSI_t[:, 0] = psi0

    for n in range(N - 1):
        PSI_t[:, n + 1] = LU.solve(U1.dot(PSI_t[:, n]))
        if print_norm:
            print(n, np.trapz(np.abs(PSI_t[:, n + 1])**2, x))

    return PSI_t


def SplitStepFourier(psi0, V, x, dt, N=100, print_norm=False):
    Nx = x.size
    dx = x[1] - x[0]

    k = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    Evol_T = np.exp(-1j * (k**2 * 0.5) * dt)
    Evol_V_half = np.exp(-1j * V * 0.5 * dt)

    PSI_t = np.zeros((Nx, N), dtype=complex)
    PSI_t[:, 0] = psi0
    psi = psi0.astype(complex)

    for n in range(N - 1):
        psi = Evol_V_half * psi
        psi = np.fft.ifft(Evol_T * np.fft.fft(psi))
        psi = Evol_V_half * psi
        PSI_t[:, n + 1] = psi
        if print_norm:
            print(n, np.trapz(np.abs(psi)**2, x))

    return PSI_t


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
        filename=filename, frame_skip=frame_skip, fps=fps
    )
