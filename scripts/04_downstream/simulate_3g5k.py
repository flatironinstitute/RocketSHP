#%%
"""simulate_3g5k.ipynb
"""

import sys

import pandas
from loguru import logger
from openmm import LangevinIntegrator, MonteCarloBarostat, Platform
from openmm import unit as u
from openmm.app import (
    PME,
    CharmmPsfFile,
    DCDReporter,
    ForceField,
    HBonds,
    Modeller,
    PDBFile,
    PDBReporter,
    Simulation,
    StateDataReporter,
)

# from openmm.unit import nanometer, picosecond, femtosecond, bar, kelvin
from rocketshp import config
from rocketshp.utils import seed_everything

#%% File paths and constants

REPLICATE_SEED = int(sys.argv[1])
# REPLICATE_SEED = 1

RAW_PDB_FILE = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_tetramer.pdb"
PSF_FILE = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_protein_autopsf.psf"
PDB_FILE = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_protein_autopsf.pdb"

DCD_OUTPUT = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_sim_r{REPLICATE_SEED}.dcd"
PDB_OUTPUT = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_sim_r{REPLICATE_SEED}.pdb"
SCALAR_OUTPUT = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_scalars_r{REPLICATE_SEED}.csv"
LOG_OUTPUT = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/3g5k_r{REPLICATE_SEED}.log"

USE_CUDA = True

INIT_PDB = f"{config.PROCESSED_DATA_DIR}/3g5k_simulation/init_r{REPLICATE_SEED}.pdb"
TEMP = 300.0 * u.kelvin
PRESSURE = 1.0 * u.bar

TIME_STEP = 2.0 * u.femtosecond
TOTAL_TIME = 100 * u.nanosecond
REPORT_EVERY = 10 * u.picosecond
N_STEPS = int(TOTAL_TIME / TIME_STEP)

#%% Configure logger.

seed_everything(REPLICATE_SEED)

# Remove any existing handlers
logger.remove()

# Add both handlers at once
logger.configure(
    handlers=[
        {"sink": sys.stdout, "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"},
        {"sink": LOG_OUTPUT, "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"}
    ]
)

#%% Download structure.

from biotite.database import rcsb

pdb = rcsb.fetch('3G5K', "pdb").read()

with open(RAW_PDB_FILE,'w+') as f:
    f.write(pdb)

#%%
## Miro magic happens here-- fill in more details later
## Meeting notes:
# this has weird cofactors (BB2, CO) that seem pretty reliant
# applying symmetry groups -- select A and B and then rotate them around (VMD symmetry plugin)
# charmm-gui online
# applying symmetry in vmd?

#%% Preparing solvated system.

# Load in the structure and add water molecules following a particular forcefield
logger.info("Loading structure and adding water molecules")
psf = CharmmPsfFile(PSF_FILE)
pdb = PDBFile(PDB_FILE)

modeller = Modeller(psf.topology, pdb.positions)
# forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
forcefield = ForceField('charmm36.xml', 'charmm36/tip3p-pme-f.xml')

modeller.addHydrogens(forcefield)
modeller.addSolvent(forcefield, model="tip3p", padding=1.0 * u.nanometer)

# Forcefield together with topology is used to create openmm system containing all the forces acting on atoms.
logger.info("Creating OpenMM system")
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, constraints=HBonds)

integrator = LangevinIntegrator(TEMP,
                                1 / u.picosecond,
                                TIME_STEP)

system.addForce(MonteCarloBarostat(PRESSURE,
                                   TEMP))

#%% Define the simulation

# Get the CUDA platform
if USE_CUDA:
    platform = Platform.getPlatformByName('CUDA')  # or 'OpenCL' if CUDA isn't available
else:
    platform = Platform.getPlatformByName('OpenCL')

# Build simulation and minimize energy
logger.info("Building simulation and minimizing energy")
simulation = Simulation(modeller.topology, system, integrator, platform)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy(maxIterations=100)
positions = simulation.context.getState(getPositions=True).getPositions()

with open(INIT_PDB, "w") as f:
    PDBFile.writeFile(simulation.topology, positions, f)

#%% Run a simulation starting from a minimized structure

reporter_interval = int(REPORT_EVERY / TIME_STEP)
simulation.reporters = []
simulation.reporters.append(DCDReporter(DCD_OUTPUT, reporter_interval))
simulation.reporters.append(PDBReporter(PDB_OUTPUT, reporter_interval))
simulation.reporters.append(
    StateDataReporter(f"{LOG_OUTPUT}.progress", 10 * reporter_interval, step=True, temperature=True, elapsedTime=True)
)
simulation.reporters.append(
    StateDataReporter(
        SCALAR_OUTPUT,
        reporter_interval,
        step=True,
        time=True,
        potentialEnergy=True,
        totalEnergy=True,
        temperature=True,
    )
)

logger.info(f"Running simulation for {N_STEPS} steps...")
simulation.step(N_STEPS)

#%% Download the pdb's and visualize in PyMOL!

df = pandas.read_csv(SCALAR_OUTPUT)
df.plot(kind="line", x="Time (ps)", y="Potential Energy (kJ/mole)")
