from argparse import ArgumentParser
from pathlib import Path
from time import time

import numpy as np
import schnetpack as spk
import schnetpack.transform as trn
from agox.environments import Environment
from agox.generators import RandomGenerator
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator

import ray

ray.init(num_cpus=16)

parser = ArgumentParser()
parser.add_argument("-i", "--index", type=int, default=1)
parser.add_argument("-sn", "--sn", type=int, default=44-1)
parser.add_argument("-o", "--o", type=int, default=48-2)

args = parser.parse_args()

seed = args.index

np.random.seed(seed)

##############################################################################
# Calculator
##############################################################################
pot_path = "/home/roenne/documents/dss-examples/dss_examples/SnO-PtSn/potentials/data/all_v1/0/best_inference_model"

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
# System & general settings:
##############################################################################

path = "/home/roenne/documents/dss-examples/dss_examples/SnO-PtSn/templates/2layer_4x4.traj"
template = read(path)
template *= (2, 2, 1)

template.set_pbc([True, True, True])

confinement_cell = template.get_cell()
confinement_cell[2, 2] = 6
confinement_corner = np.array([0, 0, 3])

environment = Environment(
    template=template,
    symbols=f"Sn{args.sn}O{args.o}",
    confinement_cell=confinement_cell,
    confinement_corner=confinement_corner,
    box_constraint_pbc=[True, True, False],
    print_report=False,
)

##############################################################################
# Search Settings:
##############################################################################

random_generator = RandomGenerator(
    **environment.get_confinement(),
    may_nucleate_at_several_places=True,
)

n_batch = 16
total_num_atoms = len(template) + args.sn + args.o

##############################################################################
# Let get the show running!
##############################################################################
@ray.remote
def generate():
    return random_generator.get_candidates(sampler=None, environment=environment)[0]

trajectory = Trajectory(f"generated.traj", "w")

while True:
    t1 = time()
    structures=[]
    while len(structures) < n_batch:
        gets = ray.get([generate.remote() for _ in range(n_batch)])
        gets = [g for g in gets if g is not None]

        for structure in gets:
            if structure is not None:
                structure = Atoms(
                    symbols=structure.symbols,
                    positions=structure.positions,
                    cell=structure.cell,
                    pbc=[True, True, True],
                )
                structures.append(structure)

    gen_time = (time() - t1)/n_batch

    fixed_atoms_mask = np.hstack([np.arange(len(template))+i*total_num_atoms for i in range(len(structures))])    
    dyn = spk.interfaces.ASEBatchwiseLBFGS(calculator=calc, atoms=structures, logfile=None, fixed_atoms_mask=fixed_atoms_mask)
    t1 = time()
    dyn.run(fmax=0.1, steps=1000)
    relax_time = (time() - t1)/n_batch
    
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


