"""
3D Flow Over a Sphere: Centerline Profiles
=================================================
Simulates flow over a voxelized sphere and extracts 
velocity and pressure profiles along the mid-axis.
Plots directly to screen using Matplotlib.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import time
import matplotlib.pyplot as plt

print(f"JAX devices: {jax.devices()}")

# Parameters
NX, NY, NZ = 120, 48, 48  # Channel dimensions
OMEGA = 1.0
TAU = 1.0
NU = 1./6.
DELTA_P = 0.00005
RHO_IN = 1.0
RHO_OUT = 1.0 - DELTA_P * 3.0
MAX_STEPS = 5000
PRINT_EVERY = 500
Q = 19

print(f"Grid: {NX}x{NY}x{NZ}  omega={OMEGA}  nu={NU:.6f}  dP={DELTA_P}")

# D3Q19 Lattice Constants
C = np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],
    [1,1,0],[-1,-1,0],[1,-1,0],[-1,1,0],[1,0,1],[-1,0,-1],
    [1,0,-1],[-1,0,1],[0,1,1],[0,-1,-1],[0,1,-1],[0,-1,1]], dtype=np.int32)
W = np.array([1./3.]+[1./18.]*6+[1./36.]*12)
OPP = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17])

cx4 = jnp.array(C[:,0]).reshape(Q,1,1,1); cy4 = jnp.array(C[:,1]).reshape(Q,1,1,1)
cz4 = jnp.array(C[:,2]).reshape(Q,1,1,1); w4 = jnp.array(W).reshape(Q,1,1,1)
cx2 = jnp.array(C[:,0]).reshape(Q,1,1);   cy2 = jnp.array(C[:,1]).reshape(Q,1,1)
cz2 = jnp.array(C[:,2]).reshape(Q,1,1);   w2 = jnp.array(W).reshape(Q,1,1)
opp = jnp.array(OPP)

POS_X = jnp.array([1,7,9,11,13]); NEG_X = jnp.array([2,8,10,12,14])
ZERO_X = jnp.array([0,3,4,5,6,15,16,17,18])

# ---------------------------------------------------------
# GEOMETRY: Single Sphere
# ---------------------------------------------------------
print("\nGenerating Sphere geometry...")
geo = np.zeros((NX, NY, NZ), dtype=int)

R = 10.0
CX, CY, CZ = 30, NY//2, NZ//2  

X, Y, Z = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
dist_sq = (X - CX)**2 + (Y - CY)**2 + (Z - CZ)**2
geo[dist_sq <= R**2] = 1  # Bounce-back sphere

is_fluid = jnp.array(geo==0)[None]
is_bb = jnp.array(geo==1)[None]
is_nd = jnp.array(geo==2)[None]

# BC masks for Zou-He (Periodic in Y and Z, pressure driven in X)
inm = np.ones((NY,NZ), bool)
outm = np.ones((NY,NZ), bool)
inlet_mask = jnp.array(inm)[None]
outlet_mask = jnp.array(outm)[None]

# ---------------------------------------------------------
# LBM CORE FUNCTIONS
# ---------------------------------------------------------
def eq3d(rho, ux, uy, uz):
    usq = ux*ux + uy*uy + uz*uz
    cu = cx4*ux[None] + cy4*uy[None] + cz4*uz[None]
    return w4 * rho[None] * (1 + 3*cu + 4.5*cu*cu - 1.5*usq[None])

def eq2d(rho, ux, uy, uz):
    usq = ux*ux + uy*uy + uz*uz
    cu = cx2*ux[None] + cy2*uy[None] + cz2*uz[None]
    return w2 * rho[None] * (1 + 3*cu + 4.5*cu*cu - 1.5*usq[None])

def macro(f):
    rho = jnp.sum(f, 0)
    ux = jnp.sum(f*cx4, 0) / rho
    uy = jnp.sum(f*cy4, 0) / rho
    uz = jnp.sum(f*cz4, 0) / rho
    return rho, ux, uy, uz

def stream(f):
    out = jnp.zeros_like(f)
    for q in range(Q):
        s = jnp.roll(f[q], C[q,0], 0)
        s = jnp.roll(s, C[q,1], 1)
        s = jnp.roll(s, C[q,2], 2)
        out = out.at[q].set(s)
    return out

def zou_he_inlet(f):
    s = f[:,0,:,:]
    rho_in = RHO_IN * jnp.ones((NY,NZ))
    ux_in = 1.0 - (jnp.sum(s[ZERO_X],0) + 2*jnp.sum(s[NEG_X],0)) / rho_in
    feq = eq2d(rho_in, ux_in, jnp.zeros((NY,NZ)), jnp.zeros((NY,NZ)))
    return f.at[:,0,:,:].set(jnp.where(inlet_mask, feq, s))

def zou_he_outlet(f):
    s = f[:,-1,:,:]
    rho_out = RHO_OUT * jnp.ones((NY,NZ))
    ux_out = -1.0 + (jnp.sum(s[ZERO_X],0) + 2*jnp.sum(s[POS_X],0)) / rho_out
    feq = eq2d(rho_out, ux_out, jnp.zeros((NY,NZ)), jnp.zeros((NY,NZ)))
    return f.at[:,-1,:,:].set(jnp.where(outlet_mask, feq, s))

@jit
def one_step(f, is_fluid, is_bb, is_nd):
    _, ux, uy, uz = macro(f)
    rho = jnp.sum(f, 0)
    feq = eq3d(rho, ux, uy, uz)
    f_coll = jnp.where(is_bb, f[opp], jnp.where(is_nd, f, f - (f - feq) / TAU))
    f_str = stream(f_coll)
    f_str = zou_he_inlet(f_str)
    f_str = zou_he_outlet(f_str)
    return f_str

# ---------------------------------------------------------
# INITIALIZATION & MAIN LOOP
# ---------------------------------------------------------
print("Initialising...")
x = np.arange(NX, dtype=np.float64)
rho0 = jnp.array(np.broadcast_to((1.0 - DELTA_P*3/(NX-1)*x)[:,None,None], (NX,NY,NZ)).copy())
z_zero = jnp.zeros((NX, NY, NZ))
f = eq3d(rho0, z_zero, z_zero, z_zero)

print("JIT compiling...")
t0 = time.time()
f = one_step(f, is_fluid, is_bb, is_nd)
f.block_until_ready()
print(f"JIT: {time.time()-t0:.1f}s")

print(f"\nRunning {MAX_STEPS} steps...")
t0 = time.time()

for step in range(1, MAX_STEPS + 1):
    f = one_step(f, is_fluid, is_bb, is_nd)
    if step % PRINT_EVERY == 0:
        f.block_until_ready()
        rho, ux, uy, uz = macro(f)
        mu = float(jnp.sum(ux) / (NX * NY * NZ))
        print(f"  step {step:5d}  <u_x>={mu:.6e}")

# ---------------------------------------------------------
# PLOTTING CENTERLINE PROFILES (Matplotlib, Interactive)
# ---------------------------------------------------------
print("\nExtracting mid-axis profiles...")
rho, ux, uy, uz = macro(f)

# Extract 1D arrays along the X-axis through the center of the sphere
u_mag_mid = np.sqrt(np.array(ux)[:, CY, CZ]**2 + np.array(uy)[:, CY, CZ]**2 + np.array(uz)[:, CY, CZ]**2)
p_mid = np.array(rho)[:, CY, CZ] / 3.0 
x_coords = np.arange(NX)

# Create the plot
plt.style.use('dark_background') # Remove this line if you prefer a white background
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Velocity
ax1.plot(x_coords, u_mag_mid, 'c-', linewidth=2, label='Velocity Magnitude |u|')
ax1.axvspan(CX - R, CX + R, color='gray', alpha=0.4, label='Solid Sphere')
ax1.set_ylabel('Velocity |u|', fontsize=12)
ax1.set_title('Flow Profiles Along Central Axis (Y=NY/2, Z=NZ/2)', fontsize=14)
ax1.grid(True, alpha=0.2)
ax1.legend()

# Plot Pressure
ax2.plot(x_coords, p_mid, 'y-', linewidth=2, label='Pressure (P = ρ/3)')
ax2.axvspan(CX - R, CX + R, color='gray', alpha=0.4, label='Solid Sphere')
ax2.set_ylabel('Pressure', fontsize=12)
ax2.set_xlabel('X Coordinate (Streamwise)', fontsize=12)
ax2.grid(True, alpha=0.2)
ax2.legend()

plt.tight_layout()
plt.show()  # Display interactively without saving to disk
