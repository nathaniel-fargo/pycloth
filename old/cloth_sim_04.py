"""
cloth_sim_rt.py  (interactive v5 – stable)
-----------------------------------------
* adaptive sub‑stepping for stability
* guards against NaN/Inf energies
* minimal axes, live scrolling energy, two‑axis wind sliders
"""

import argparse, numpy as np, matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from collections import deque
g = 9.81
GRAV = np.array([0, 0, -g])

@dataclass
class SpringArray:
    i1: np.ndarray; i2: np.ndarray
    rest: np.ndarray; k: float

class Cloth:
    def __init__(self, nx, nz, ℓ, m, k, ζ):
        self.nx, self.nz, self.n = nx, nz, nx*nz
        self.m, self.k, self.ζ = m, k, ζ
        xs = np.linspace(0,(nx-1)*ℓ,nx); zs = np.linspace(0,-(nz-1)*ℓ,nz)
        xx, zz = np.meshgrid(xs,zs)
        self.pos = np.c_[xx.ravel(), np.zeros(self.n), zz.ravel()]
        self.vel = np.zeros_like(self.pos)
        self.anchor = np.arange(nx); self.pos0 = self.pos.copy()
        i1,i2,rest = [],[],[]
        for j in range(nz):
            for i in range(nx):
                idx=j*nx+i
                if i<nx-1: i1+=idx,; i2+=idx+1,; rest+=ℓ,
                if j<nz-1: i1+=idx,; i2+=idx+nx,; rest+=ℓ,
                if i<nx-1 and j<nz-1: i1+=idx,; i2+=idx+nx+1,; rest+=ℓ*np.sqrt(2),
                if i>0 and j<nz-1:     i1+=idx,; i2+=idx+nx-1,; rest+=ℓ*np.sqrt(2),
        self.spring = SpringArray(np.array(i1),np.array(i2),np.array(rest),k)
        self.z0 = self.pos[:,2].min()

    # vectorised energies
    def energies(self):
        ke = 0.5*self.m*np.sum(self.vel**2)
        ge = self.m*g*np.sum(self.pos[:,2]-self.z0)
        d = self.pos[self.spring.i2]-self.pos[self.spring.i1]
        stretch = np.linalg.norm(d,axis=1)-self.spring.rest
        ee = 0.5*self.spring.k*np.sum(stretch**2)
        return ke,ge,ee

    # one small step
    def _step(self, dt, wind):
        f = np.broadcast_to(self.m*GRAV+wind*self.m, self.pos.shape).copy()
        d = self.pos[self.spring.i2]-self.pos[self.spring.i1]
        L = np.linalg.norm(d,axis=1,keepdims=True)
        dir = np.divide(d,L,out=np.zeros_like(d),where=L!=0)
        fs = self.spring.k*(L-self.spring.rest[:,None])*dir
        np.add.at(f,self.spring.i1,fs); np.add.at(f,self.spring.i2,-fs)
        f += -self.ζ*self.vel
        self.vel += (f/self.m)*dt
        self.pos += self.vel*dt
        self.pos[self.anchor] = self.pos0[self.anchor]
        self.vel[self.anchor] = 0

    # adaptive wrapper
    def step(self, DT, wind):
        dt_safe = 0.4*np.sqrt(self.m/self.k/4)   # degree≈4
        n = max(1,int(np.ceil(DT/dt_safe)))
        sub = DT/n
        for _ in range(n): self._step(sub, wind)

    def grid(self):
        g = self.pos.reshape(self.nz,self.nx,3); return g[:,:,0],g[:,:,1],g[:,:,2]

# ---------- main ----------
def main():
    p=argparse.ArgumentParser(description='Stable cloth'); add=p.add_argument
    add('--nx',type=int,default=20); add('--nz',type=int,default=20)
    add('--rest_len',type=float,default=0.15)
    add('--mass',type=float,default=0.05); add('--k',type=float,default=500)
    add('--damping',type=float,default=0.05)
    add('--dt',type=float,default=0.006); add('--fps',type=int,default=60)
    add('--window',type=float,default=10)
    A=p.parse_args(); cloth=Cloth(A.nx,A.nz,A.rest_len,A.mass,A.k,A.damping)
    win_frames=int(A.window/A.dt)
    tB,keB,geB,eeB,totB = (deque(maxlen=win_frames) for _ in range(5))
    fig=plt.figure(figsize=(10,5)); gs=fig.add_gridspec(1,2,width_ratios=[3,2],wspace=.25)
    ax=fig.add_subplot(gs[0],projection='3d'); ax.view_init(20,0)
    for ax_ in (ax.xaxis,ax.yaxis,ax.zaxis): ax_.set_ticks([])
    ax.set_xlabel('+x'); ax.set_ylabel('+y'); ax.set_zlabel('+z')
    ax.set_xlim(0,(A.nx-1)*A.rest_len); span=max(A.nx-1,A.nz-1)*A.rest_len/2
    ax.set_ylim(-span,span); ax.set_zlim(-(A.nz-1)*A.rest_len,0.1)
    e=ax.plot_surface(*cloth.grid(),color='cornflowerblue',edgecolor='grey',lw=.3,alpha=.9)
    axE=fig.add_subplot(gs[1]); axE.set_xlabel('t (s)'); axE.set_ylabel('E (J)')
    (lke,)=axE.plot([],[],lw=1,label='KE'); (lge,)=axE.plot([],[],lw=1,label='GPE')
    (lee,)=axE.plot([],[],lw=1,label='Elastic'); (ltot,)=axE.plot([],[],lw=1.3,label='Total')
    axE.set_xlim(0,A.window); axE.set_ylim(0,1); axE.legend(fontsize=8)
    s_axx=fig.add_axes([.15,.03,.55,.015]); s_axy=fig.add_axes([.15,.0,.55,.015])
    sx=Slider(s_axx,'Wind +x',-15,15,valinit=0); sy=Slider(s_axy,'Wind +y',-15,15,valinit=0)
    frame=0
    def upd(_):
        nonlocal e, frame
        wind=np.array([sx.val,sy.val,0.0]); frame+=1; t=frame*A.dt
        cloth.step(A.dt,wind)
        ke,ge,ee=cloth.energies(); tot=ke+ge+ee
        if not np.isfinite(tot): print('💥 simulation blew up (NaN/Inf)'); plt.close(); return
        tB.append(t); keB.append(ke); geB.append(ge); eeB.append(ee); totB.append(tot)
        t0=tB[0]; τ=np.array(tB)-t0
        for line,data in zip((lke,lge,lee,ltot),(keB,geB,eeB,totB)): line.set_data(τ,data)
        axE.set_xlim(0,max(A.window,τ[-1])); axE.set_ylim(0,max(totB)*1.2+1e-8)
        e.remove(); e=ax.plot_surface(*cloth.grid(),color='cornflowerblue',
                                     edgecolor='grey',lw=.3,alpha=.9)
        return e,lke,lge,lee,ltot
    FuncAnimation(fig,upd,interval=1000/A.fps,blit=False); plt.show()

if __name__=='__main__': main()