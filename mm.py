#!/usr/bin/env python
"""
Pure MM molecular dynamics script
=======================================================================
Usage:
    python mm.py

Input files:
    - protein.pdb  (protein structure)
    - ligand.sdf  (ligand structure)

Output files (./md_output/):
    - preminimized.pdb     (solvated, before minimization)
    - minimized.pdb        (after energy minimization)
    - nvt_equil.pdb        (after NVT equilibration)
    - npt_equil.pdb        (after NPT equilibration)
    - simulation.xtc       (production trajectory)
    - simulation.log       (production log)
    - checkpoint.chk       (checkpoint)
"""

import os
import sys
import time
import logging
import pathlib

import openmm
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule as OFFMolecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

# ============================================================================
# User configuration
# ============================================================================

PROTEIN_PDB = "protein.pdb"
LIGAND_SDF = "ligand.sdf"
OUTPUT_DIR = pathlib.Path("./md_output")

# Force fields
PROTEIN_FF = "amber/protein.ff14SB.xml"
WATER_FF = "amber/tip3p_standard.xml"
SMALL_MOL_FF = "openff_unconstrained-2.0.0.offxml"

# Solvation
SOLVENT_PADDING = 1.0 * unit.nanometer
IONIC_STRENGTH = 0.15 * unit.molar

# Integrator
TIMESTEP = 2.0 * unit.femtoseconds
TEMPERATURE = 300.0 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
PRESSURE = 1.0 * unit.bar
BAROSTAT_FREQ = 25  # steps

# Simulation lengths
MINIMIZATION_STEPS = 5000
NVT_EQUIL_LENGTH = 0.1 * unit.nanosecond
NPT_EQUIL_LENGTH = 1.0 * unit.nanosecond
PRODUCTION_LENGTH = 1.0 * unit.nanosecond

# Output intervals
CHECKPOINT_INTERVAL = 0.01 * unit.nanosecond  # = 5000 steps @ 2fs
LOG_INTERVAL = 5000  # steps

# Compute platform (None for auto, "CUDA", "OpenCL", "CPU")
PLATFORM_NAME = None

# ============================================================================
# Logging setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MM-SimpleMD")


def steps_from_length(length, timestep):
    """Convert a time length to number of steps."""
    return int(length / timestep)


def get_platform(name=None):
    """Get the OpenMM compute platform."""
    if name is not None:
        platform = openmm.Platform.getPlatformByName(name)
    else:
        platform = None  # OpenMM will auto-select fastest
    return platform


def main():
    t_start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # =====================================================================
    # 1. Load molecules
    # =====================================================================
    logger.info("Loading protein and ligand...")
    protein_pdb = app.PDBFile(PROTEIN_PDB)
    ligand_mol = OFFMolecule.from_file(LIGAND_SDF, allow_undefined_stereo=True)

    # =====================================================================
    # 2. Set force fields (protein FF + small-molecule SMIRNOFF)
    # =====================================================================
    logger.info("Setting force fields...")
    forcefield = app.ForceField(PROTEIN_FF, WATER_FF)
    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=[ligand_mol], forcefield=SMALL_MOL_FF
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)

    # =====================================================================
    # 3. Build solvated system
    # =====================================================================
    logger.info("Building solvated system...")
    modeller = app.Modeller(protein_pdb.topology, protein_pdb.positions)

    # Add ligand
    lig_topology = ligand_mol.to_topology().to_openmm()
    lig_positions = ligand_mol.conformers[0].to_openmm()
    modeller.add(lig_topology, lig_positions)

    # Add solvent
    modeller.addSolvent(
        forcefield,
        padding=SOLVENT_PADDING,
        ionicStrength=IONIC_STRENGTH,
        model="tip3p",
    )

    topology = modeller.getTopology()
    positions = modeller.getPositions()

    logger.info(f"Total atoms: {topology.getNumAtoms()}")
    logger.info(f"Total residues: {topology.getNumResidues()}")

    # Save solvated structure
    with open(OUTPUT_DIR / "preminimized.pdb", "w") as f:
        app.PDBFile.writeFile(topology, positions, f, keepIds=True)
    logger.info("Saved solvated structure: preminimized.pdb")

    # =====================================================================
    # 4. Create MM system
    # =====================================================================
    logger.info("Creating OpenMM system...")
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.0 * unit.amu,  # HMR for larger timesteps
    )

    # =====================================================================
    # 5. Create integrator and Simulation
    # =====================================================================
    integrator = openmm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    platform = get_platform(PLATFORM_NAME)

    if platform is not None:
        simulation = app.Simulation(topology, system, integrator, platform)
    else:
        simulation = app.Simulation(topology, system, integrator)

    simulation.context.setPositions(positions)

    logger.info(f"Using platform: {simulation.context.getPlatform().getName()}")

    # =====================================================================
    # 6. Energy minimization
    # =====================================================================
    logger.info(f"Energy minimization ({MINIMIZATION_STEPS} steps)...")
    e_before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"  Potential energy before minimization: {e_before}")

    simulation.minimizeEnergy(maxIterations=MINIMIZATION_STEPS)

    e_after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"  Potential energy after minimization: {e_after}")

    min_state = simulation.context.getState(getPositions=True)
    with open(OUTPUT_DIR / "minimized.pdb", "w") as f:
        app.PDBFile.writeFile(topology, min_state.getPositions(), f, keepIds=True)
    logger.info("Saved minimized structure: minimized.pdb")

    # =====================================================================
    # 7. NVT equilibration
    # =====================================================================
    nvt_steps = steps_from_length(NVT_EQUIL_LENGTH, TIMESTEP)
    logger.info(f"NVT equilibration: {NVT_EQUIL_LENGTH} ({nvt_steps} steps)...")

    simulation.context.setVelocitiesToTemperature(TEMPERATURE)
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            reportInterval=LOG_INTERVAL,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
            progress=True,
            totalSteps=nvt_steps,
            remainingTime=True,
        )
    )

    simulation.step(nvt_steps)

    nvt_state = simulation.context.getState(getPositions=True)
    with open(OUTPUT_DIR / "equil_nvt.pdb", "w") as f:
        app.PDBFile.writeFile(topology, nvt_state.getPositions(), f, keepIds=True)
    logger.info("NVT equilibration complete, saved: equil_nvt.pdb")

    # Clear reporters
    simulation.reporters.clear()

    # =====================================================================
    # 8. NPT equilibration
    # =====================================================================
    npt_steps = steps_from_length(NPT_EQUIL_LENGTH, TIMESTEP)
    logger.info(f"NPT equilibration: {NPT_EQUIL_LENGTH} ({npt_steps} steps)...")

    # Add barostat
    barostat = openmm.MonteCarloBarostat(PRESSURE, TEMPERATURE, BAROSTAT_FREQ)
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            reportInterval=LOG_INTERVAL,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            progress=True,
            totalSteps=npt_steps,
            remainingTime=True,
        )
    )

    simulation.step(npt_steps)

    npt_state = simulation.context.getState(getPositions=True)
    with open(OUTPUT_DIR / "equil_npt.pdb", "w") as f:
        app.PDBFile.writeFile(topology, npt_state.getPositions(), f, keepIds=True)
    logger.info("NPT equilibration complete, saved: equil_npt.pdb")

    # Clear reporters
    simulation.reporters.clear()

    # =====================================================================
    # 9. Production run
    # =====================================================================
    prod_steps = steps_from_length(PRODUCTION_LENGTH, TIMESTEP)
    chk_steps = steps_from_length(CHECKPOINT_INTERVAL, TIMESTEP)
    logger.info(f"Production run: {PRODUCTION_LENGTH} ({prod_steps} steps)...")

    simulation.reporters.append(
        app.StateDataReporter(
            str(OUTPUT_DIR / "simulation.log"),
            reportInterval=LOG_INTERVAL,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            reportInterval=LOG_INTERVAL,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            speed=True,
            progress=True,
            totalSteps=prod_steps,
            remainingTime=True,
        )
    )
    simulation.reporters.append(
        app.XTCReporter(
            str(OUTPUT_DIR / "simulation.xtc"),
            reportInterval=chk_steps,
            enforcePeriodicBox=True,
        )
    )
    simulation.reporters.append(
        app.CheckpointReporter(
            str(OUTPUT_DIR / "checkpoint.chk"),
            reportInterval=chk_steps,
        )
    )

    simulation.step(prod_steps)

    # =====================================================================
    # Done
    # =====================================================================
    t_end = time.time()
    elapsed = t_end - t_start
    logger.info(f"Simulation complete! Total time: {elapsed:.1f} s ({elapsed/3600:.2f} h)")
    logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
