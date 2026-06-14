#!/usr/bin/env python
"""
ML/MM hybrid MD script
=====================================================================================================
Supported ML force fields:
    "aimnet2", "torchmdnet", "aceff-1.0", "aceff-1.1", "aceff-2.0"

Workflow:
    1. MM minimization (remove bad geometry)
    2. ML/MM light minimization
    3. ML/MM NVT equilibration
    4. ML/MM NPT equilibration
    5. ML/MM production run

Usage:
    python mlmm.py

Input files:
    - protein.pdb  (protein structure)
    - ligand.sdf   (ligand structure)

Output files (./md_output/):
    - preminimized.pdb     (solvated, before minimization)
    - minimized_mm.pdb     (after MM minimization)
    - minimized_mlmm.pdb   (after ML/MM minimization)
    - equil_nvt.pdb        (after NVT equilibration)
    - equil_npt.pdb        (after NPT equilibration)
    - simulation.xtc       (production trajectory, full atoms)
    - simulation.log       (production log)
    - checkpoint.chk       (checkpoint)
"""

import os
import sys
import time
import math
import logging
import pathlib
from typing import List, Optional


import openmm
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule as OFFMolecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmml import MLPotential


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


# ML force-field settings
# Options: "aimnet2", "aceff-2.0" etc.
ML_MODEL = "aceff-2.0"
# Ligand residue name candidates (OpenFF defaults to "UNK", sometimes "LIG" or "MOL")
LIGAND_RESNAMES = ["UNK", "LIG", "MOL"]


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
MM_MINIMIZATION_STEPS = 5000      # MM minimization steps (more, to relax geometry)
MLMM_MINIMIZATION_STEPS = 5000    # ML/MM minimization steps (fewer, for refinement)
NVT_EQUIL_LENGTH = 0.1 * unit.nanosecond
NPT_EQUIL_LENGTH = 1.0 * unit.nanosecond
PRODUCTION_LENGTH = 1.0 * unit.nanosecond


# Output intervals
CHECKPOINT_INTERVAL = 0.01 * unit.nanosecond
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
logger = logging.getLogger("ML/MM-StepwiseMD")




def steps_from_length(length, timestep):
    """Convert a time length to number of steps."""
    return int(length / timestep)




def get_platform(name=None):
    """Get the OpenMM compute platform."""
    if name is not None:
        platform = openmm.Platform.getPlatformByName(name)
    else:
        platform = None
    return platform




def get_ligand_atom_indices(
    topology: app.Topology,
    ligand_resnames: List[str] = ["UNK", "LIG", "MOL"],
) -> List[int]:
    """
    Get ligand atom indices from the topology.


    Parameters
    ----------
    topology : app.Topology
        OpenMM topology
    ligand_resnames : list of str
        Candidate ligand residue names


    Returns
    -------
    list of int
        Ligand atom indices
    """
    indices = []
    for residue in topology.residues():
        if residue.name in ligand_resnames:
            for atom in residue.atoms():
                indices.append(atom.index)
    return indices




def create_mlmm_system(
    topology: app.Topology,
    mm_system: openmm.System,
    ml_atom_indices: List[int],
    ml_model: str = "aceff-2.0",
    interpolate: bool = False,
    **kwargs,
) -> openmm.System:
    """
    Convert an MM system to an ML/MM mixed system.


    Bonded forces within the ligand are replaced by the ML potential, while
    nonbonded ligand-solvent/protein interactions remain MM.


    Parameters
    ----------
    topology : app.Topology
        OpenMM topology
    mm_system : openmm.System
        Original MM system
    ml_atom_indices : list of int
        Atom indices to be modeled with ML
    ml_model : str
        ML model name
    interpolate : bool
        Whether to enable lambda interpolation (for FEP; False for plain MD)


    Returns
    -------
    openmm.System
        ML/MM mixed system
    """
    potential = MLPotential(ml_model)
    mixed_system = potential.createMixedSystem(
        topology=topology,
        system=mm_system,
        atoms=ml_atom_indices,
        removeConstraints=True,
        interpolate=interpolate,
        charge=0,                     #only for aceff and aimnet2
        multiplicity=1,               #only for aimnet2
        **kwargs,
    )
    return mixed_system




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


    # Get ligand atom indices
    ml_atom_indices = get_ligand_atom_indices(topology, LIGAND_RESNAMES)
    logger.info(f"Ligand atom count: {len(ml_atom_indices)}")
    if len(ml_atom_indices) > 0:
        logger.info(f"Ligand atom indices: {ml_atom_indices[:5]}...")


    if len(ml_atom_indices) == 0:
        raise ValueError(f"No ligand atoms found! Residue names searched: {LIGAND_RESNAMES}")


    # =====================================================================
    # 4. Step 1: MM minimization
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 1: MM force-field minimization")
    logger.info("=" * 60)
    
    # Create pure MM system
    mm_system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        hydrogenMass=1.0 * unit.amu,
    )


    # MM minimization
    integrator_mm = openmm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    platform = get_platform(PLATFORM_NAME)


    if platform is not None:
        simulation_mm = app.Simulation(topology, mm_system, integrator_mm, platform)
    else:
        simulation_mm = app.Simulation(topology, mm_system, integrator_mm)


    simulation_mm.context.setPositions(positions)
    logger.info(f"Using platform: {simulation_mm.context.getPlatform().getName()}")


    # Energy check
    initial_pe = simulation_mm.context.getState(getEnergy=True).getPotentialEnergy()
    pe_value = initial_pe.value_in_unit(unit.kilojoule_per_mole)
    if not math.isfinite(pe_value):
        raise RuntimeError(
            f"Initial energy is {initial_pe}. Check for atom overlaps or bad ligand placement."
        )
    logger.info(f"MM initial potential energy: {initial_pe}")


    # MM minimization
    logger.info(f"MM minimization ({MM_MINIMIZATION_STEPS} steps)...")
    simulation_mm.minimizeEnergy(maxIterations=MM_MINIMIZATION_STEPS)


    e_after_mm = simulation_mm.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"MM potential energy after minimization: {e_after_mm}")


    # Save MM-minimized structure
    mm_state = simulation_mm.context.getState(getPositions=True)
    minimized_positions = mm_state.getPositions()
    
    with open(OUTPUT_DIR / "minimized_mm.pdb", "w") as f:
        app.PDBFile.writeFile(topology, minimized_positions, f, keepIds=True)


    # Release MM simulation
    del simulation_mm, integrator_mm


    # =====================================================================
    # 5. Step 2: Switch to ML/MM system
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Switch to ML/MM mixed system")
    logger.info("=" * 60)
    
    # Create ML/MM mixed system
    logger.info(f"Creating ML/MM mixed system (model: {ML_MODEL})...")
    mlmm_system = create_mlmm_system(
        topology=topology,
        mm_system=mm_system,
        ml_atom_indices=ml_atom_indices,
        ml_model=ML_MODEL,
        interpolate=False,
    )
    logger.info("ML/MM mixed system created successfully!")


    # Create new simulation
    integrator = openmm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
    
    if platform is not None:
        simulation = app.Simulation(topology, mlmm_system, integrator, platform)
    else:
        simulation = app.Simulation(topology, mlmm_system, integrator)


    # Set coordinates from MM minimization
    simulation.context.setPositions(minimized_positions)


    # Light ML/MM minimization
    logger.info(f"ML/MM light minimization ({MLMM_MINIMIZATION_STEPS} steps)...")
    simulation.minimizeEnergy(maxIterations=MLMM_MINIMIZATION_STEPS)
    
    e_after_mlmm = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    logger.info(f"ML/MM potential energy after minimization: {e_after_mlmm}")


    # Save ML/MM-minimized structure
    mlmm_state = simulation.context.getState(getPositions=True)
    with open(OUTPUT_DIR / "minimized_mlmm.pdb", "w") as f:
        app.PDBFile.writeFile(topology, mlmm_state.getPositions(), f, keepIds=True)


    # =====================================================================
    # 6. Step 3: NVT equilibration
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 3: NVT equilibration")
    logger.info("=" * 60)
    
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
    logger.info("NVT equilibration complete")


    simulation.reporters.clear()


    # =====================================================================
    # 7. Step 4: NPT equilibration
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 4: NPT equilibration")
    logger.info("=" * 60)
    
    npt_steps = steps_from_length(NPT_EQUIL_LENGTH, TIMESTEP)
    logger.info(f"NPT equilibration: {NPT_EQUIL_LENGTH} ({npt_steps} steps)...")


    barostat = openmm.MonteCarloBarostat(PRESSURE, TEMPERATURE, BAROSTAT_FREQ)
    mlmm_system.addForce(barostat)
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
    logger.info("NPT equilibration complete")


    simulation.reporters.clear()


    # =====================================================================
    # 8. Step 5: Production run
    # =====================================================================
    logger.info("=" * 60)
    logger.info("Step 5: Production run")
    logger.info("=" * 60)
    
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
    
    # Trajectory output: all atoms
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
    logger.info("=" * 60)
    logger.info("Simulation complete!")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f} s ({elapsed/3600:.2f} h)")
    logger.info(f"Output directory: {OUTPUT_DIR.resolve()}")
    logger.info(f"Trajectory file: simulation.xtc")




if __name__ == "__main__":
    main()
