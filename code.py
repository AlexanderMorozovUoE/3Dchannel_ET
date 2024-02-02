import numpy as np
import h5py
from dedalus import public as de
from dedalus.extras import flow_tools
import time
import math

from numpy.random import default_rng

rng = default_rng()


import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)

Lx, Ly, Lz = (10.0, 2., 10.0)
nx, ny, nz = (256, 1024, 256)
Tmax = 20000.0
Reynolds = 0.01
Wi = 150.
beta = 0.8
dt = 0.0025

PTT = 0.001
diffusivity = 0.00005


# Create bases and domain
z_basis = de.Fourier('z', nz, interval=(0, Lz), dealias=3/2)
x_basis = de.Fourier('x', nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Chebyshev('y',ny, interval=(-1, 1), dealias=3/2)
domain = de.Domain([z_basis, x_basis, y_basis], grid_dtype=np.float64, mesh=(128,128))

problem = de.IVP(domain, variables=['p','u','v','w','uy','vy','wy',\
    'cxx', 'cxy', 'cxz', 'cyy', 'cyz', 'czz', \
    'Dcxx', 'Dcxy', 'Dcxz', 'Dcyy', 'Dcyz', 'Dczz'])

problem.parameters['Remin1'] = 1.0/Reynolds
problem.parameters['Wimin1'] = 1.0/Wi
problem.parameters['beta'] = beta
problem.parameters['Lx'] = Lx
problem.parameters['Ly'] = Ly
problem.parameters['Lz'] = Lz
problem.parameters['kappa'] = diffusivity
problem.parameters['PTT'] = PTT

problem.substitutions['TrC'] = "(PTT*Wimin1)*(cxx+cyy+czz)"
problem.substitutions['Gc'] = "Wimin1*(1-3*PTT)"
problem.substitutions['ux'] = "dx(u)"
problem.substitutions['vx'] = "dx(v)"
problem.substitutions['wx'] = "dx(w)"
problem.substitutions['uz'] = "dz(u)"
problem.substitutions['vz'] = "dz(v)"
problem.substitutions['wz'] = "dz(w)"
problem.substitutions['Lap(A,Ay)'] = "dx(dx(A)) + dy(Ay) + dz(dz(A))"
problem.substitutions['Adv(A,Ay)'] = "u*dx(A) + v*Ay + w*dz(A)"

problem.substitutions['Fxx'] = "- Adv(cxx,Dcxx) + Gc - TrC*cxx \
                                + 2.0*(cxx*ux + cxy*uy + cxz*uz)"

problem.substitutions['Fyy'] = "- Adv(cyy,Dcyy) + Gc - TrC*cyy \
                                + 2.0*(cxy*vx + cyy*vy + cyz*vz)"

problem.substitutions['Fzz'] = "- Adv(czz,Dczz) + Gc - TrC*czz \
                                + 2.0*(cxz*wx + cyz*wy + czz*wz)"

problem.substitutions['Fxy'] = "- Adv(cxy,Dcxy) - TrC*cxy \
                                + cxx*vx - cxy*wz + cxz*vz + cyy*uy + cyz*uz"

problem.substitutions['Fxz'] = "- Adv(cxz,Dcxz) - TrC*cxz \
                                + cxx*wx + cxy*wy - cxz*vy + cyz*uy + czz*uz"

problem.substitutions['Fyz'] = "- Adv(cyz,Dcyz) - TrC*cyz \
                                + cxy*wx + cxz*vx + cyy*wy - cyz*ux + czz*vz"

# Ux equation
problem.add_equation("dt(u) + dx(p) - beta*Remin1*Lap(u,uy) \
        - Remin1*Wimin1*(1.0-beta)*( dx(cxx) + dy(cxy) + dz(cxz) ) \
        = - (u*ux + v*uy + w*uz) + 2*Remin1")

# Uy equation
problem.add_equation("dt(v) + dy(p) - beta*Remin1*Lap(v,vy) \
        - Remin1*Wimin1*(1.0-beta)*( dx(cxy) + dy(cyy) + dz(cyz) ) \
        = - (u*vx + v*vy + w*vz)")

# Uz equation
problem.add_equation("dt(w) + dz(p) - beta*Remin1*Lap(w,wy) \
        - Remin1*Wimin1*(1.0-beta)*( dx(cxz) + dy(cyz) + dz(czz) ) \
        = - (u*wx + v*wy + w*wz)")

# Incompressibility
problem.add_equation("dx(u) + vy + dz(w) = 0")

# C equations
problem.add_equation("dt(cxx) + Gc*cxx - kappa*Lap(cxx,Dcxx) - TrC = Fxx")
problem.add_equation("dt(cyy) + Gc*cyy - kappa*Lap(cyy,Dcyy) - TrC = Fyy")
problem.add_equation("dt(czz) + Gc*czz - kappa*Lap(czz,Dczz) - TrC = Fzz")
problem.add_equation("dt(cxy) + Gc*cxy - kappa*Lap(cxy,Dcxy)       = Fxy")
problem.add_equation("dt(cxz) + Gc*cxz - kappa*Lap(cxz,Dcxz)       = Fxz")
problem.add_equation("dt(cyz) + Gc*cyz - kappa*Lap(cyz,Dcyz)       = Fyz")

# Derivatives
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")
problem.add_equation("wy - dy(w) = 0")

problem.add_equation("Dcxx - dy(cxx) = 0")
problem.add_equation("Dcxy - dy(cxy) = 0")
problem.add_equation("Dcxz - dy(cxz) = 0")
problem.add_equation("Dcyy - dy(cyy) = 0")
problem.add_equation("Dcyz - dy(cyz) = 0")
problem.add_equation("Dczz - dy(czz) = 0")

# Boundary Conditions
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")

problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0) or (nz != 0)")
problem.add_bc("left(p) = 0", condition="(nx == 0) and (nz == 0)")

problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0")

# Stress Boundary Conditions
problem.add_equation("left( dt(cxx) + Gc*cxx - TrC ) = left( Fxx )")
problem.add_equation("left( dt(cyy) + Gc*cyy - TrC ) = left( Fyy )")
problem.add_equation("left( dt(czz) + Gc*czz - TrC ) = left( Fzz )")
problem.add_equation("left( dt(cxy) + Gc*cxy       ) = left( Fxy )")
problem.add_equation("left( dt(cxz) + Gc*cxz       ) = left( Fxz )")
problem.add_equation("left( dt(cyz) + Gc*cyz       ) = left( Fyz )")

problem.add_equation("right( dt(cxx) + Gc*cxx - TrC ) = right( Fxx )")
problem.add_equation("right( dt(cyy) + Gc*cyy - TrC ) = right( Fyy )")
problem.add_equation("right( dt(czz) + Gc*czz - TrC ) = right( Fzz )")
problem.add_equation("right( dt(cxy) + Gc*cxy       ) = right( Fxy )")
problem.add_equation("right( dt(cxz) + Gc*cxz       ) = right( Fxz )")
problem.add_equation("right( dt(cyz) + Gc*cyz       ) = right( Fyz )")

####################
# Build solver

solver = problem.build_solver(de.timesteppers.SBDF4)
logger.info('Solver built')

solver.stop_sim_time = Tmax
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

####################
# Load configuration

#write, olddt = solver.load_state('init.h5', 0)


####################
# Add a random perturbation to Cxx

cxx = solver.state['cxx']; Dcxx = solver.state['Dcxx']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=3/2)
slices = domain.dist.grid_layout.slices(scales=3/2)
noise = rng.standard_normal(gshape)[slices]

cxx['g']  += 625.*noise
cxx.differentiate('y',out=Dcxx)

####################

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10.0, max_writes=5)
snapshots.add_system(solver.state)

ke = solver.evaluator.add_file_handler('energies', sim_dt=0.1, max_writes=500)
ke.add_task("(15./4.)*(1/2)*integ(u*u+v*v+w*w,'x','y','z')/(Lx*Ly*Lz)", name='KE')

# Runtime monitoring properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property("(15./4.)*(1/2)*integ(u*u+v*v+w*w,'x','y','z')/(Lx*Ly*Lz)", name='q')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            max_energy = flow.max('q')
            logger.info('Kinetic energy = %f' %max_energy)
            if math.isnan(max_energy):
                raise Exception('NaN occured in max_energy. stopping.')

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    run_time = end_time - start_time
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %run_time)
    logger.info('Run time: %f cpu-hr' %(run_time/60/60*domain.dist.comm_cart.size))
    DOF = nx * ny * nz
    logger.info('Speed: %.2e DOF-iters/cpu-sec' %(DOF*solver.iteration/run_time/domain.dist.comm.size))
