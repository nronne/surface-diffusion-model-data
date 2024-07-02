import torch
import pytorch_lightning as pl
import numpy as np

from ase.io import read

from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory
from dss.helpers import get_diffusion_model, sample
from argparse import ArgumentParser
from glob import glob

from ase.optimize import BFGS
from ase.constraints import FixAtoms

from time import time

from chgnet.model import CHGNetCalculator



parser = ArgumentParser()
parser.add_argument('--index', '-i', type=int, default=0)
parser.add_argument("-ag", "--ag", type=int, default=29)
parser.add_argument("-o", "--o", type=int, default=22)

args = parser.parse_args()

pl.seed_everything(args.index)


##############################################################################
# Calculator
##############################################################################

calc = CHGNetCalculator(use_device='cuda')
            
##############################################################################
# Diffusion Model
##############################################################################

# Load dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, _ = get_diffusion_model(cutoff=6.0, beta_max=2.0)

checkpoint_path = 'model.ckpt'
state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
model.load_state_dict(state_dict)

model.to(device)

model.verbose = False
##############################################################################
# System & general settings:
##############################################################################

path = '/home/roenne/documents/dss-examples/dss_examples/AgO/templates/empty_stripe_2layer.traj'
template = read(path)

template.set_pbc([True, True, True])

z_confinement = torch.tensor(np.array([2.5, 7.8]))

ag_placed, o_placed = 0, 0
symbols = ['Ag']*(args.ag-ag_placed) + ['O']*(args.o - o_placed)
eta = 1e-4
n_batch = 16


total_num_atoms = len(template) + args.ag + args.o - ag_placed - o_placed

trajectory = Trajectory(f"ag{args.ag}o{args.o}.traj", "w")

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
            c = FixAtoms(indices=np.arange(len(template)))
            structure.set_constraint(c)
            structures.append(structure)

    gen_time = (t2 - t1)/len(structures)

    for structure in structures:
        structure.calc = calc
        dyn = BFGS(structure, logfile=None)
        
        t1 = time()
        dyn.run(fmax=0.05, steps=1000)
        relax_time = (time() - t1)
        
        e, f = structure.get_potential_energy(), structure.get_forces()
        print(f"Energy: {e:.2f}, Relax: {dyn.get_number_of_steps()} steps in {relax_time:.2f} s, Generation Time: {gen_time:.2f} s, Total Time: {gen_time + relax_time:.2f} s", flush=True)
        structure.calc = SinglePointCalculator(structure, energy=e, forces=f)
        structure.info["steps"] = dyn.get_number_of_steps()
        structure.info["relax_time"] = relax_time
        structure.info["gen_time"] = gen_time
        trajectory.write(structure)
    

