import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator


def compute_peak_positions(PSI0, x, a, xlim_min, xlim_max):
    rho = np.abs(PSI0)**2
    x_normalized = x / a
    mask = (x_normalized >= xlim_min) & (x_normalized <= xlim_max)
    
    Nt = PSI0.shape[1]
    X0 = np.zeros(Nt)
    window_half = int(5 * a / (x[1] - x[0]))
    
    for t in range(Nt):
        rho_t = rho[:, t].copy()
        rho_t[~mask] = 0
        peak_idx = np.argmax(rho_t)
        
        i_start = max(0, peak_idx - window_half)
        i_end = min(len(x), peak_idx + window_half + 1)
        
        rho_window = rho_t[i_start:i_end]
        x_window = x[i_start:i_end]
        
        if np.sum(rho_window) > 0:
            X0[t] = np.sum(rho_window * x_window) / np.sum(rho_window) / a
        else:
            X0[t] = x[peak_idx] / a
    
    return X0


def setup_axes_style(axes):
    for ax in axes:
        ax.set_facecolor('#EAEAF2')
        ax.grid(True, color='white', linewidth=1.2)
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_axisbelow(True)


def create_bloch_animation(PSI0, Enk, phi_nk, Vx, x, k_1d_bz, params, 
                           filename='bloch_anim.mp4', frame_skip=10, fps=25):
    a, dt, Tbloch = params['a'], params['dt'], params['Tbloch']
    n0, nbnds = params['n0'], params['nbnds']
    xlim_min, xlim_max = -30, 60
    
    X0 = compute_peak_positions(PSI0, x, a, xlim_min, xlim_max)
    kwht = np.abs(phi_nk.conj() @ PSI0).T
    
    deep_blue = '#4C72B0'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(8.0, 3.0), dpi=300)
    ax_band, ax_wave = fig.subplot_mosaic([[0, 1]], width_ratios=[1, 5]).values()
    setup_axes_style([ax_band, ax_wave])

    ax_band.axvline(x=0, lw=0.8, ls='--', color='gray')
    for ii in range(nbnds):
        ax_band.plot(k_1d_bz, Enk[:, ii], color='black', lw=1.2, zorder=1)
    scat = ax_band.scatter(k_1d_bz, Enk[:, n0], s=kwht[0] / 3,
                           lw=0.5, color=deep_blue, facecolor='white', zorder=2)
    ax_band.set_xlim(-0.5, 0.5)
    ax_band.set_xticks([-0.5, 0, 0.5], [r'$-\frac{\pi}{a}$', '0', r'$\frac{\pi}{a}$'])
    ax_band.set_ylim(-0.05, 0.25)
    ax_band.set_ylabel('Energy [a.u.]')

    ax_pot = ax_wave.twinx()
    line_wfc, = ax_wave.plot(x/a, np.abs(PSI0[:, 0]), lw=1.2, color=deep_blue)
    line_x0, = ax_wave.plot([X0[0]], [0], ls='none', marker='o', ms=8, mew=1.2, mfc='white', mec='black')
    ax_pot.plot(x/a, Vx, lw=1.0, color='black', alpha=0.8)
    
    ax_wave.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax_wave.set_xlim(xlim_min, xlim_max)
    ax_wave.set_ylim(-0.02, 0.25)
    ax_pot.set_ylim(-0.80, 0.20)
    ax_wave.set_xlabel(r'$x / a$', labelpad=5)
    ax_wave.set_ylabel(r'$|\psi(x)|$ [a.u.]', labelpad=5)
    ax_pot.set_ylabel(r'$V(x)$ [a.u.]', labelpad=5)
    for spine in ax_pot.spines.values():
        spine.set_color('white')

    time_text = ax_wave.text(0.98, 0.10, '', transform=ax_wave.transAxes, 
                             ha='right', family='monospace', fontsize='small')
    plt.tight_layout()

    NSW = PSI0.shape[1]
    n_frames = NSW // frame_skip
    interval = 1000 // fps
    
    def update(frame):
        idx = min(frame * frame_skip, NSW - 1)
        line_wfc.set_ydata(np.abs(PSI0[:, idx]))
        line_x0.set_xdata([X0[idx]])
        scat.set_sizes(kwht[idx] / 3)
        time_text.set_text(r'$t={:6.2f}\,\tau_B$'.format(idx * dt / Tbloch))
        return line_wfc, line_x0, scat, time_text

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    ani.save(filename, writer='ffmpeg', dpi=150)
    plt.show()
    print(f"Animation saved to {filename} ({n_frames} frames, ~{n_frames/fps:.1f}s)")
