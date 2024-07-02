import torch
import pytorch_lightning as pl
import numpy as np
from scipy.spatial.distance import cdist

from ase.io import read, write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from dss.helpers import get_diffusion_model, sample
from argparse import ArgumentParser
from glob import glob
import schnetpack.transform as trn

from ase.optimize import BFGS
from ase.constraints import FixAtoms

from time import time

import schnetpack as spk
import schnetpack.transform as trn


parser = ArgumentParser()
parser.add_argument('--index', '-i', type=int, default=0)
parser.add_argument("-sn", "--sn", type=int, default=44-1)
parser.add_argument("-o", "--o", type=int, default=48-2)

args = parser.parse_args()

pl.seed_everything(args.index)


##############################################################################
# Calculator
##############################################################################
pot_path = 'potential'

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.MatScipyNeighborList(cutoff=6.0), device="cuda")
calc = spk.interfaces.BatchwiseCalculator(
    model=pot_path,
    atoms_converter=converter,
    device='cuda',
    energy_key="energy",
    force_key="forces",
    energy_unit="eV",
    position_unit="Ang",
)

##############################################################################
# Diffusion Model
##############################################################################

# Load dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = get_diffusion_model(cutoff=6.0, beta_max=4.0)

checkpoint_path = 'model.ckpt'
state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
model.load_state_dict(state_dict)

model.to(device)


model.verbose = False
##############################################################################
# System & general settings:
##############################################################################

path = '/home/roenne/documents/dss-examples/dss_examples/SnO-PtSn/templates/2layer_4x4.traj'
template = read(path)
template *= (2,2,1)
template.set_pbc([True, True, True])

z_confinement = torch.tensor(np.array([2.5, 9.8]))

symbols = ['Sn']*args.sn + ['O']*args.o
eta = 1e-4
n_batch = 16



total_num_atoms = len(template) + args.sn + args.o

trajectory = Trajectory(f"generated.traj", "w")

while True:
    t1 = time()
    atoms = sample(model, n_batch, template, symbols, z_confinement, eta=eta, num_steps=1000, postrelax_steps=0)
    t2 = time()

    structures = []
    for structure in atoms:
        d = structure.get_all_distances(mic=True)
        d[np.diag_indices_from(d)] = 1e10
        if d.min() < 1.0:
            continue
        else:
            structures.append(structure)

    gen_time = (t2 - t1)/len(structures)
            
    fixed_atoms_mask = np.hstack([np.arange(len(template))+i*total_num_atoms for i in range(len(structures))])
    dyn = spk.interfaces.ASEBatchwiseLBFGS(calculator=calc, atoms=structures, logfile=None, fixed_atoms_mask=fixed_atoms_mask)
    
    t1 = time()                                
    dyn.run(fmax=0.1, steps=1000)
    relax_time = (time() - t1)/len(structures)
    
    structures, _ = dyn.get_relaxation_results()
    
    for structure in structures:
        structure.calc = calc
        e, f = structure.get_potential_energy(), structure.get_forces()
        print(f"Energy: {e[0]:.2f}, Relax: {dyn.get_number_of_steps()} steps in {relax_time:.2f} s, Generation Time: {gen_time:.2f} s, Total Time: {gen_time + relax_time:.2f} s", flush=True)
        structure.calc = SinglePointCalculator(structure, energy=e[0], forces=f)
        structure.info["steps"] = dyn.get_number_of_steps()
        structure.info["relax_time"] = relax_time
        structure.info["gen_time"] = gen_time
        trajectory.write(structure)
    

