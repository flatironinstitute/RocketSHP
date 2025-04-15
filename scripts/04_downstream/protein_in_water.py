#!/usr/bin/env python
# coding: utf-8
# python -m openmm.testInstallation
# %% Imports
from openmm.app import (
   Simulation, 
   Modeller,
   ForceField,
   StateDataReporter,
   CheckpointReporter,
   PME,
   HBonds,
   PDBFile,
)
from openmm import (
    Platform,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
)
from openmm.unit import (
    bar,
    kelvin,
    nanometer,
    nanosecond,
    picosecond,
)
from pdbfixer import PDBFixer
from subset_xtc_reporter import XTCReporter

from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from pathlib import Path

import biotite.structure as bs
from biotite.structure.io.pdb import PDBFile as bsPDBFile
from biotite.database import rcsb
from biotite.interface import openmm as omm_if

# %% Define inputs
import argparse

platforms = []
num_platforms = Platform.getNumPlatforms()
for i in range(num_platforms):
    platforms.append(Platform.getPlatform(i).getName())

parser = argparse.ArgumentParser(description="Run OpenMM simulation on a protein structure")
parser.add_argument(
    "--pdb-path",
    type=Path,
    required=True,
    help="Path to the PDB file of the protein structure",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Path to the output directory",
)
parser.add_argument(
    "--run-id",
    type=str,
    default="simulation",
    help="Run ID for the simulation [simulation]",
)
parser.add_argument(
    "--temperature","-T",
    type=float,
    default=300,
    help="Temperature in Kelvin [300]",
)
parser.add_argument(
    "--pressure","-p",
    type=float,
    default=1,
    help="Pressure in bar [1]",
)
parser.add_argument(
    "--time",
    type=float,
    default=100,
    help="Total simulation time in nanoseconds [100]",
)
parser.add_argument(
    "--step-size","-s",
    type=float,
    default=0.002,
    help="Step size in picoseconds [0.002]",
)
parser.add_argument(
    "--save-every",
    type=int,
    default=10,
    help="Save trajectory every N picoseconds [10]"
)
parser.add_argument(
    "--checkpoint-scale",
    type=int,
    default=500,
    help="Save checkpoint every N times the save interval [500]",
)
parser.add_argument(
    "--no-nvt-eq",
    action="store_true",
    help="Run NVT equilibration",
)
parser.add_argument(
    "--no-npt-eq",
    action="store_true",
    help="Run NPT equilibration",
)
parser.add_argument(
    "--no-plot",
    action="store_true",
    help="Do not plot results",
)
parser.add_argument(
    "--device", "-d",
    type=str,
    default="CUDA",
    choices=platforms,
    help="Device to run the simulation on [CUDA]",
)
parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Random seed for the simulation [-1 = no seed]",
)

args = parser.parse_args()
args.nvt_eq = not args.no_nvt_eq
args.npt_eq = not args.no_npt_eq


pdb_path = args.pdb_path
assert pdb_path.exists(), f"PDB file {pdb_path} does not exist."
output_dir = args.output_dir
output_dir.mkdir(parents=True, exist_ok=True)
run_id = args.run_id

temperature = args.temperature * kelvin
pressure = args.pressure * bar
step_size = args.step_size * picosecond
save_every = args.save_every * picosecond
checkpoint_scale = args.checkpoint_scale
total_time = args.time * nanosecond
device = args.device
seed = args.seed

save_steps = int(save_every / step_size)

NVT_EQ_TIME = 200 * picosecond
NPT_EQ_TIME = 1 * nanosecond

logger.info(f"Running simulation on {pdb_path} at {temperature} and {pressure} for {total_time}.")

# %% Derive parameters
# Convert time to steps
total_steps = int(total_time / step_size)

nvt_steps = 0
npt_steps = 0

if args.nvt_eq:
    nvt_steps = int(NVT_EQ_TIME / step_size) # 10 ns
if args.npt_eq:
    npt_steps = int(NPT_EQ_TIME / step_size) # 90 ns

# %% Load structure and topology and fix

fixer = PDBFixer(filename=str(pdb_path))
fixer.findMissingResidues()
fixer.findNonstandardResidues()
fixer.replaceNonstandardResidues()
fixer.removeHeterogens(True)
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(7.0)

# Write topology to a file called top.pdb
PDBFile.writeFile(fixer.topology, fixer.positions, file=str(output_dir / f"{run_id}_top.pdb"))

# pdb_fi = PDBFile(str(pdb_path))

# %% Use Modeller to Fix Structure

# Specify the forcefield
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

#Preprocessing the pdb file.
# modeller = Modeller(omm_top, omm_pos)
modeller = Modeller(fixer.topology, fixer.positions)
modeller.deleteWater()

protein_atoms = []
for atom in modeller.topology.atoms():
    if atom.residue.name in PDBFile._standardResidues:
        protein_atoms.append(atom.index)

residues=modeller.addHydrogens(forcefield)
modeller.addSolvent(forcefield, padding=1.0*nanometer)

# %% Create the simulation
# Here we define some forcefield settings such as the nonbonded method
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)

# Define the integrator. The Langevin integrator is also a thermostat
integrator = LangevinMiddleIntegrator(temperature, 1/picosecond, step_size)
if seed != -1:
    integrator.setRandomNumberSeed(seed)

platform = Platform.getPlatformByName(device) 
logger.info(f"Using platform: {platform.getName()}")
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)

# Print state information to the screen every 1000 steps
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
        potentialEnergy=True, temperature=True, volume=True,
        elapsedTime=True, remainingTime=True, totalSteps=nvt_steps+npt_steps+total_steps))

# %% Minimize energy
logger.info("Minimizing energy...")
simulation.minimizeEnergy()

# %% NVT Equilibration
if args.nvt_eq:
    logger.info(f"Running NVT equilibration for {NVT_EQ_TIME / nanosecond} nanoseconds ({nvt_steps} steps)")
    simulation.step(nvt_steps)


# %% NPT Equilibiration
if args.npt_eq:
    logger.info(f"Running NPT equilibration for {NPT_EQ_TIME / nanosecond} nanoseconds ({npt_steps} steps)")
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(npt_steps)

# %% Set up reporters

# Write trajectory to a file called traj.xtc
logger.warning(f"Only writing protein atoms to the trajectory.")
xtc_reporter = XTCReporter(str(output_dir / f"{run_id}_traj.xtc"), save_steps, atomSubset=protein_atoms)
simulation.reporters.append(xtc_reporter)

# Print the same info to a log file
state_data_reporter = StateDataReporter(
    str(output_dir / f"{run_id}_log.txt"),
    save_steps,
    step=True,
    potentialEnergy=True,
    temperature=True,
    volume=True
)
simulation.reporters.append(state_data_reporter)

# Save checkpoint every 100x save steps
checkpoint_reporter = CheckpointReporter(str(output_dir / f"{run_id}_checkpoint.chk"), checkpoint_scale*save_steps)
simulation.reporters.append(checkpoint_reporter)

# %% Production run
logger.info(f"Running production run for {total_time / nanosecond} nanoseconds ({total_steps} steps)")
simulation.context.reinitialize(preserveState=True)
simulation.step(total_steps)

# %% Plot results
if not args.no_plot:
    logger.info("Plotting results...")
    data = np.loadtxt(output_dir / f"{run_id}_log.txt", delimiter=',')
    step = data[:,0]
    potential_energy = data[:,1]
    temperature = data[:,2]
    volume = data[:,3]

    # Potential Energy
    plt.figure(figsize=(10, 10)) 
    plt.subplot(3, 1, 1) 
    plt.plot(step, potential_energy, color='b', linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Potential Energy (kJ/mol)")

    # Temperature
    plt.subplot(3, 1, 2) 
    plt.plot(step, temperature, color='r', linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Temperature (K)")

    # Volume
    plt.subplot(3, 1, 3) 
    plt.plot(step, volume, color='g', linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Volume (nm³)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{run_id}_md_plot.png", dpi=300)
    plt.close()
