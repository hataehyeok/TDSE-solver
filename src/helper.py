import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator

def create_bloch_animation(PSI0, Enk, phi_nk, Vx, x, k_1d_bz, params, filename='bloch_anim.mp4'):
    """
    Bloch Oscillation 시뮬레이션 결과를 받아 애니메이션을 생성 및 저장합니다.
    
    Parameters:
    - PSI0: 시간 의존 파동함수 (Nx, Nt)
    - Enk: 밴드 에너지 (Nk, Nbands)
    - phi_nk: Bloch 기저 함수 (Nk, Nx)
    - Vx: 포텐셜 (Nx,)
    - x: 위치 배열 (Nx,)
    - k_1d_bz: k-공간 배열 (Nk,)
    - params: 시뮬레이션 파라미터 딕셔너리 {'a', 'dt', 'Tbloch', 'n0', 'nbnds'}
    - filename: 저장할 파일명
    """
    
    # --- 1. 데이터 전처리 (Vectorized) ---
    # 파라미터 언패킹
    a, dt, Tbloch = params['a'], params['dt'], params['Tbloch']
    n0, nbnds = params['n0'], params['nbnds']
    
    # 위치 기댓값 <x> 계산
    rho = np.abs(PSI0)**2
    norm = np.sum(rho, axis=0)
    X0 = np.sum(rho * x[:, None], axis=0) / norm / a
    
    # k-공간 투영 (Loop 제거 및 최적화)
    # phi_nk.conj() shape: (Nk, Nx), PSI0 shape: (Nx, Nt) -> (Nk, Nt)
    # Transpose하여 (Nt, Nk) 형태로 변환
    kwht = np.abs(phi_nk.conj() @ PSI0).T 
    
    # --- 2. 그래프 스타일 설정 ---
    deep_blue = '#4C72B0'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(8.0, 3.0), dpi=300)
    ax_band, ax_wave = fig.subplot_mosaic([[0, 1]], width_ratios=[1, 5]).values()
    
    # 공통 스타일 적용
    for ax in [ax_band, ax_wave]:
        ax.set_facecolor('#EAEAF2')
        ax.grid(True, color='white', linewidth=1.2)
        for spine in ax.spines.values(): spine.set_color('white')
        ax.set_axisbelow(True)

    # (왼쪽) 밴드 구조
    ax_band.axvline(x=0, lw=0.8, ls='--', color='gray')
    for ii in range(nbnds):
        ax_band.plot(k_1d_bz, Enk[:, ii], color='black', lw=1.2, zorder=1)
    
    # 현재 k상태 산점도 초기화
    scat = ax_band.scatter(k_1d_bz, Enk[:, n0], s=kwht[0] / 3,
                           lw=0.5, color=deep_blue, facecolor='white', zorder=2)
    
    ax_band.set_xlim(-0.5, 0.5)
    ax_band.set_xticks([-0.5, 0, 0.5], [r'$-\frac{\pi}{a}$', '0', r'$\frac{\pi}{a}$'])
    ax_band.set_ylim(-0.05, 0.25)
    ax_band.set_ylabel('Energy [a.u.]')

    # (오른쪽) 파동함수 및 포텐셜
    ax_pot = ax_wave.twinx()
    
    # 데이터 플롯 초기화
    line_wfc, = ax_wave.plot(x/a, np.abs(PSI0[:, 0]), lw=1.2, color=deep_blue)
    line_x0, = ax_wave.plot([X0[0]], [0], ls='none', marker='o', 
                            ms=8, mew=1.2, mfc='white', mec='black')
    
    ax_pot.plot(x/a, Vx, lw=1.0, label=r'$V(x)$', color='black', alpha=0.8)
    
    # 축 설정
    ax_wave.xaxis.set_minor_locator(AutoMinorLocator(n=5))
    ax_wave.set_xlim(-30, 60)
    ax_wave.set_ylim(-0.02, 0.25)
    ax_pot.set_ylim(-0.80, 0.20)
    
    ax_wave.set_xlabel(r'$x / a$', labelpad=5)
    ax_wave.set_ylabel(r'$|\psi(x)|$ [a.u.]', labelpad=5)
    ax_pot.set_ylabel(r'$V(x)$ [a.u.]', labelpad=5)
    
    # 오른쪽 축 스타일링
    for spine in ax_pot.spines.values(): spine.set_color('white')

    # 타임스탬프 텍스트
    time_text = ax_wave.text(0.98, 0.10, '', transform=ax_wave.transAxes, 
                             ha='right', family='monospace', fontsize='small')

    plt.tight_layout()

    # --- 3. 애니메이션 업데이트 함수 ---
    NSW = PSI0.shape[1]
    
    def update(frame):
        idx = frame * 10
        if idx >= NSW: idx = NSW - 1
        
        # 데이터 업데이트
        line_wfc.set_ydata(np.abs(PSI0[:, idx]))
        line_x0.set_xdata([X0[idx]])
        scat.set_sizes(kwht[idx] / 3)
        
        # 텍스트 업데이트
        t_val = idx * dt / Tbloch
        time_text.set_text(r'$t={:6.2f}\,\tau_B$'.format(t_val))
        
        return line_wfc, line_x0, scat, time_text

    # 애니메이션 생성 및 저장
    ani = animation.FuncAnimation(fig, update, frames=NSW//10, interval=40, blit=True)
    ani.save(filename, writer='ffmpeg', dpi=300)
    plt.show()
    print(f"Animation saved to {filename}")