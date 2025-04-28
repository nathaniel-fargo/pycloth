"""
cloth_sim_rt.py (interactive v7 – animation fix)
------------------------------------------------
* nonlocal surf inside update so surface actually refreshes each frame
* energy plot now updates regardless of user interaction
* minor speed tweak: reuse Poly3DCollection via set_verts if available
"""

import argparse, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
from collections import deque
from dataclasses import dataclass

g = 9.81
GRAV = np.array([0, 0, -g])

@dataclass
class SpringArray:
    i1: np.ndarray
    i2: np.ndarray
    rest: np.ndarray
    k: float

class Cloth:
    def __init__(self, nx, nz, rest, m, k, damp):
        self.nx, self.nz, self.n = nx, nz, nx*nz
        self.m, self.k, self.damp = m, k, damp
        xs = np.linspace(0,(nx-1)*rest,nx); zs = np.linspace(0,-(nz-1)*rest,nz)
        xx, zz = np.meshgrid(xs, zs)
        self.pos = np.c_[xx.ravel(), np.zeros(self.n), zz.ravel()]
        self.vel = np.zeros_like(self.pos)
        self.anchor = np.arange(nx); self.p0 = self.pos.copy()

        i1,i2,rl = [],[],[]
        for j in range(nz):
            for i in range(nx):
                idx=j*nx+i
                if i<nx-1: i1.append(idx); i2.append(idx+1); rl.append(rest)
                if j<nz-1: i1.append(idx); i2.append(idx+nx); rl.append(rest)
                if i<nx-1 and j<nz-1: i1.append(idx); i2.append(idx+nx+1); rl.append(rest*np.sqrt(2))
                if i>0 and j<nz-1: i1.append(idx); i2.append(idx+nx-1); rl.append(rest*np.sqrt(2))
        self.spring = SpringArray(np.array(i1),np.array(i2),np.array(rl),k)
        self.z0=self.pos[:,2].min()

    def _step(self, dt, wind):
        f = np.broadcast_to(self.m*GRAV+wind*self.m, self.pos.shape).copy()
        d = self.pos[self.spring.i2]-self.pos[self.spring.i1]
        L = np.linalg.norm(d,axis=1,keepdims=True)+1e-12
        fs = self.spring.k*(L-self.spring.rest[:,None])*(d/L)
        np.add.at(f,self.spring.i1,fs); np.add.at(f,self.spring.i2,-fs)
        f += -self.damp*self.vel
        self.vel += (f/self.m)*dt
        self.pos += self.vel*dt
        self.pos[self.anchor] = self.p0[self.anchor]
        self.vel[self.anchor] = 0

    def step(self, DT, wind):
        dt_safe = 0.15*np.sqrt(self.m/self.k)
        n = max(1,int(np.ceil(DT/dt_safe))); sub=DT/n
        for _ in range(n): self._step(sub,wind)

    def energies(self):
        ke=0.5*self.m*np.sum(self.vel**2)
        ge=self.m*g*np.sum(self.pos[:,2]-self.p0[:,2])
        stretch=np.linalg.norm(self.pos[self.spring.i2]-self.pos[self.spring.i1],axis=1)-self.spring.rest
        ee=0.5*self.k*np.sum(stretch**2)
        return ke,ge,ee

    def grid(self):
        g=self.pos.reshape(self.nz,self.nx,3); return g[:,:,0],g[:,:,1],g[:,:,2]

def main():
    ap=argparse.ArgumentParser(); ad=ap.add_argument
    ad('--nx',type=int,default=20); ad('--nz',type=int,default=20)
    ad('--rest_len',type=float,default=0.15); ad('--mass',type=float,default=0.05)
    ad('--k',type=float,default=500); ad('--damping',type=float,default=0.06)
    ad('--dt',type=float,default=0.01); ad('--fps',type=int,default=45)
    ad('--window',type=float,default=10)
    A=ap.parse_args()

    cloth=Cloth(A.nx,A.nz,A.rest_len,A.mass,A.k,A.damping)
    win=int(A.window/A.dt)
    tB,keB,geB,eeB,totB=(deque(maxlen=win) for _ in range(5))

    fig=plt.figure(figsize=(10,5))
    gs=fig.add_gridspec(1,2,width_ratios=[3,2],wspace=.25)
    ax=fig.add_subplot(gs[0],projection='3d'); ax.view_init(20,0)
    for ax_ in (ax.xaxis,ax.yaxis,ax.zaxis): ax_.set_ticks([])
    ax.set_xlim(0,(A.nx-1)*A.rest_len)
    sp=max(A.nx-1,A.nz-1)*A.rest_len/2; ax.set_ylim(-sp,sp); ax.set_zlim(-(A.nz-1)*A.rest_len,0.1)
    ax.set_xlabel('+x'); ax.set_ylabel('+y'); ax.set_zlabel('+z')
    X,Y,Z=cloth.grid()
    surf=ax.plot_surface(X,Y,Z,color='cornflowerblue',edgecolor='grey',lw=.3,alpha=.9)

    axE=fig.add_subplot(gs[1]); axE.set_xlabel('t (s)'); axE.set_ylabel('E (J)')
    lke,=axE.plot([],[],lw=1,label='KE'); lge,=axE.plot([],[],lw=1,label='GPE')
    lee,=axE.plot([],[],lw=1,label='Elastic'); ltot,=axE.plot([],[],lw=1.3,label='Total')
    axE.set_xlim(0,A.window); axE.set_ylim(0,1); axE.legend(fontsize=8)

    ax_sx=fig.add_axes([.15,.03,.55,.015]); ax_sy=fig.add_axes([.15,.0,.55,.015])
    sx=Slider(ax_sx,'Wind +x',-15,15); sy=Slider(ax_sy,'Wind +y',-15,15)

    frame=0
    def update(_):
        nonlocal frame, surf
        frame+=1; t=frame*A.dt
        wind=np.array([sx.val,sy.val,0])
        cloth.step(A.dt,wind)
        ke,ge,ee=cloth.energies(); tot=ke+ge+ee
        if not np.isfinite(tot): print("Diverged"); plt.close(); return
        tB.append(t); keB.append(ke); geB.append(ge); eeB.append(ee); totB.append(tot)
        τ=np.array(tB)-tB[0]
        for line,data in zip((lke,lge,lee,ltot),(keB,geB,eeB,totB)): line.set_data(τ,data)
        axE.set_xlim(0,max(A.window,τ[-1])); axE.set_ylim(0,max(totB)*1.2+1e-6)

        # Update surface
        X,Y,Z=cloth.grid()
        surf.remove()
        surf=ax.plot_surface(X,Y,Z,color='cornflowerblue',edgecolor='grey',lw=.3,alpha=.9)
        
        return surf,lke,lge,lee,ltot

    def on_slider_change(val):
        update(None)
        fig.canvas.draw_idle()

    sx.on_changed(on_slider_change)
    sy.on_changed(on_slider_change)

    anim = FuncAnimation(fig, update, interval=1000/A.fps, blit=False, cache_frame_data=False)
    plt.show()

if __name__=='__main__':
    main()
