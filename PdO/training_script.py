from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

from dss.helpers import get_dataset, get_diffusion_model
from dss.utils.ema import EMA, EMACheckpoint


parser = ArgumentParser()
parser.add_argument('--index', '-i', type=int, default=0)
args = parser.parse_args()

pl.seed_everything(args.index)

# Load dataset
path = "...path_to_dataset"
trajs = read(path, index=":")
for t in trajs:
    e, f = t.get_potential_energy(), t.get_forces(apply_constraint=False)
    t.set_pbc([True, True, True])
    t.set_calculator(SinglePointCalculator(t, energy=e, forces=f))


model, neighbour_list = get_diffusion_model(cutoff=6.0, lr=1e-3)

# checkpoint_path = ''
# state_dict = torch.load()['state_dict']
# model.load_state_dict(state_dict)

    
dataset = get_dataset(trajs, neighbour_list, repeats=[1])


logger = pl.loggers.TensorBoardLogger(save_dir="logs")
callbacks = [
    pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    EMA(decay=0.99),
    EMACheckpoint(
        filename="{step}",
        every_n_train_steps=20_000,
        save_top_k=-1,
    ),
    EMACheckpoint(
        filename="best_{step}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    ),
]

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=callbacks,
    gradient_clip_val=1.0,
    max_time={"days": 4, "hours": 0, "minutes": 0},
    enable_progress_bar=False,

)
trainer.fit(model, datamodule=dataset)


