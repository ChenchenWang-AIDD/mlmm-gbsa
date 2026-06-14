#!/usr/bin/env python3
"""
MM-GBSA / MM-PBSA Binding Free Energy Pipeline
============================================================
Converts OpenMM MD outputs to Amber format and runs MMPBSA.py.
  - Supports both GAFF and OpenFF SMIRNOFF ligand force fields (--ligand-ff)
  - Supports GB, PB, or both methods

Inputs (auto-detected from --datadir):
  - equil_npt.pdb   : reference PDB from OpenMM (protein + ligand + ions, no water)
  - simulation.xtc  : production trajectory from OpenMM
  - ligand.sdf      : original ligand SDF file

Usage:
  mamba activate mlmm_gbsa

  # Run MM-GBSA (default):
  python gbsa.py --workdir . --method gb --igb 5

  # Run MM-PBSA:
  python gbsa.py --method pb --istrng 0.15

  # Run both:
  python gbsa.py --method both

  # Specify output directory:
  python gbsa.py --outdir ./my_gbsa_results
"""

import argparse
import os
import sys
import shutil
import subprocess
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def parse_args():
    p = argparse.ArgumentParser(
        description='MM-GBSA/PBSA pipeline for OpenMM trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Directories ---
    p.add_argument('--workdir', default='.',
                   help='Working directory (default: .)')
    p.add_argument('--datadir', default=None,
                   help='Data directory containing input files '
                        '(default: <workdir>/mlmm_output)')
    p.add_argument('--outdir', default=None,
                   help='Output directory for all generated files '
                        '(default: <workdir>/gbsa_results)')
    p.add_argument('--ligand-resname', default='UNK',
                   help='Residue name of the ligand in the PDB (default: UNK)')

    # --- Method selection ---
    p.add_argument('--method', default='gb', choices=['gb', 'pb', 'both'],
                   help='Solvation method: gb, pb, or both (default: gb)')

    # --- GB parameters ---
    p.add_argument('--igb', type=int, default=5, choices=[1, 2, 5, 7, 8],
                   help='GB model (default: 5 = OBC2)')
    p.add_argument('--saltcon', type=float, default=0.150,
                   help='Salt concentration in M for GB (default: 0.150)')

    # --- PB parameters ---
    p.add_argument('--istrng', type=float, default=0.150,
                   help='Ionic strength in M for PB (default: 0.150)')
    p.add_argument('--fillratio', type=float, default=4.0,
                   help='PB grid fill ratio (default: 4.0)')
    p.add_argument('--inp', type=int, default=2, choices=[1, 2],
                   help='PB nonpolar solvation: 1=old, 2=new (default: 2)')
    p.add_argument('--radiopt', type=int, default=0, choices=[0, 1],
                   help='PB radii optimization: 0=use prmtop radii (recommended for OpenFF), '
                        '1=PB internal radii by atom type (GAFF only) (default: 0)')
    p.add_argument('--cavity_surften', type=float, default=0.0378,
                   help='Surface tension (kcal/mol/A^2, default: 0.0378)')
    p.add_argument('--cavity_offset', type=float, default=-0.5692,
                   help='Cavity offset (kcal/mol, default: -0.5692)')

    # --- Frame selection ---
    p.add_argument('--startframe', type=int, default=1,
                   help='First frame (1-based, default: 1)')
    p.add_argument('--endframe', type=int, default=0,
                   help='Last frame (0 = all, default: 0)')
    p.add_argument('--interval', type=int, default=1,
                   help='Frame interval (default: 1)')

    # --- Force field ---
    p.add_argument('--forcefields', nargs='+',
                   default=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
                   help='OpenMM protein force field XMLs')
    p.add_argument('--extra-forcefields', nargs='*', default=[],
                   help='Additional force field XMLs')
    p.add_argument('--ligand-ff', default='openff_unconstrained-2.0.0.offxml',
                   help='Ligand force field: GAFF version (e.g. gaff-2.11) or '
                        'SMIRNOFF offxml (e.g. openff_unconstrained-2.0.0.offxml). '
                        'Must match the force field used in MD!')

    # --- Misc ---
    p.add_argument('--keep-intermediate', action='store_true',
                   help='Keep intermediate files (prmtop, inpcrd, nc)')
    p.add_argument('--decomp', action='store_true',
                   help='Run per-residue decomposition')

    return p.parse_args()


# ==============================================================================
# Step 1: Create Amber prmtop files from OpenMM topology
# ==============================================================================
def create_prmtop_files(ref_pdb_path, ligand_sdf_path, lig_resname, forcefields,
                        extra_forcefields, ligand_ff, outdir, igb=5):
    """
    Reconstruct the OpenMM System from the reference PDB + force field,
    then convert to Amber prmtop via ParmEd.

    Parameters
    ----------
    ligand_ff : str
        Either a GAFF version string (e.g. 'gaff-2.11') or an OpenFF SMIRNOFF
        offxml file (e.g. 'openff_unconstrained-2.0.0.offxml').

    Returns
    -------
    complex_prmtop, receptor_prmtop, ligand_prmtop : str
        Paths to the generated prmtop files.
    n_complex_atoms : int
    complex_atom_indices : np.ndarray
    """
    from openmm.app import PDBFile, ForceField, NoCutoff
    from openff.toolkit import Molecule
    import parmed
    from parmed.tools import strip

    print("[Step 1/4] Creating Amber topology files via OpenMM → ParmEd...")

    # --- Load ligand ---
    mol = Molecule.from_file(ligand_sdf_path)
    print(f"  Ligand: {mol.n_atoms} atoms, charge={mol.total_charge}")

    # --- Setup force field ---
    ff_files = list(forcefields) + list(extra_forcefields)
    ff = ForceField(*ff_files)

    is_smirnoff = ligand_ff.endswith('.offxml')
    if is_smirnoff:
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
        gen = SMIRNOFFTemplateGenerator(molecules=mol, forcefield=ligand_ff)
        ff.registerTemplateGenerator(gen.generator)
        print(f"  Ligand FF: OpenFF SMIRNOFF ({ligand_ff})")
    else:
        from openmmforcefields.generators import GAFFTemplateGenerator
        gen = GAFFTemplateGenerator(molecules=mol, forcefield=ligand_ff)
        ff.registerTemplateGenerator(gen.generator)
        print(f"  Ligand FF: GAFF ({ligand_ff})")

    print(f"  Protein FF: {ff_files}")

    # --- Load reference PDB ---
    pdb = PDBFile(ref_pdb_path)
    print(f"  Reference PDB: {pdb.topology.getNumAtoms()} atoms")

    # --- Create OpenMM System ---
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
    print(f"  OpenMM System: {system.getNumParticles()} particles")

    # --- Convert to ParmEd Structure ---
    struct = parmed.openmm.load_topology(
        pdb.topology, system=system, xyz=pdb.positions
    )
    print(f"  ParmEd structure: {len(struct.atoms)} atoms, "
          f"{len(struct.residues)} residues")

    # --- Identify ions/water ---
    ion_water_names = {
        'NA', 'CL', 'K', 'SOD', 'CLA', 'POT', 'MG', 'CA', 'ZN', 'FE',
        'Na+', 'Cl-', 'K+', 'Mg2+', 'Ca2+',
        'WAT', 'HOH', 'T3P', 'T4P', 'T4E', 'SPC', 'TIP3',
    }

    residues_to_strip = []
    for r in struct.residues:
        if r.name in ion_water_names:
            residues_to_strip.append(r.name)

    if residues_to_strip:
        from collections import Counter
        print(f"  Stripping: {dict(Counter(residues_to_strip))}")

    # Build complex atom indices
    complex_atom_indices = np.array([
        atom.idx for atom in struct.atoms
        if atom.residue.name not in ion_water_names
    ])

    # --- Complex prmtop ---
    strip_mask_parts = sorted(set(residues_to_strip))
    if strip_mask_parts:
        ion_mask = ':' + ','.join(strip_mask_parts)
        complex_struct = struct.copy(parmed.Structure)
        strip(complex_struct, ion_mask).execute()
    else:
        complex_struct = struct.copy(parmed.Structure)

    n_complex = len(complex_struct.atoms)
    print(f"  Complex: {n_complex} atoms, {len(complex_struct.residues)} residues")

    complex_prmtop = os.path.join(outdir, 'complex.prmtop')
    complex_struct.save(complex_prmtop, overwrite=True)
    complex_struct.save(os.path.join(outdir, 'complex.inpcrd'), overwrite=True)

    # --- Receptor prmtop ---
    rec_struct = complex_struct.copy(parmed.Structure)
    strip(rec_struct, f':{lig_resname}').execute()
    n_receptor = len(rec_struct.atoms)
    print(f"  Receptor: {n_receptor} atoms, {len(rec_struct.residues)} residues")

    receptor_prmtop = os.path.join(outdir, 'receptor.prmtop')
    rec_struct.save(receptor_prmtop, overwrite=True)
    rec_struct.save(os.path.join(outdir, 'receptor.inpcrd'), overwrite=True)

    # --- Ligand prmtop ---
    lig_res_indices = [r.idx for r in complex_struct.residues
                       if r.name == lig_resname]
    if not lig_res_indices:
        avail = sorted(set(r.name for r in complex_struct.residues))
        print(f"  ERROR: No residue '{lig_resname}' found! Available: {avail}")
        sys.exit(1)

    non_lig = sorted(set(r.name for r in complex_struct.residues
                         if r.name != lig_resname))
    lig_struct = complex_struct.copy(parmed.Structure)
    strip(lig_struct, ':' + ','.join(non_lig)).execute()
    n_ligand = len(lig_struct.atoms)
    print(f"  Ligand: {n_ligand} atoms, {len(lig_struct.residues)} residues")

    ligand_prmtop = os.path.join(outdir, 'ligand.prmtop')
    lig_struct.save(ligand_prmtop, overwrite=True)
    lig_struct.save(os.path.join(outdir, 'ligand.inpcrd'), overwrite=True)

    # --- Sanity check ---
    assert n_receptor + n_ligand == n_complex, \
        (f"Atom count mismatch: receptor({n_receptor}) + "
         f"ligand({n_ligand}) != complex({n_complex})")
    print(f"  ✓ receptor({n_receptor}) + ligand({n_ligand}) = complex({n_complex})")

    # --- Assign GB radii ---
    igb_to_radii = {1: 'mbondi', 2: 'mbondi2', 5: 'mbondi2',
                    7: 'bondi', 8: 'mbondi3'}
    radii_set = igb_to_radii.get(igb, 'mbondi2')
    print(f"  Setting GB radii: {radii_set} (for igb={igb})")

    from parmed.tools import changeRadii as ChangeRadii
    for path in [complex_prmtop, receptor_prmtop, ligand_prmtop]:
        parm = parmed.load_file(path)
        ChangeRadii(parm, radii_set).execute()
        parm.save(path, overwrite=True)

    print("  ✓ GB radii assigned")

    return (complex_prmtop, receptor_prmtop, ligand_prmtop,
            n_complex, complex_atom_indices)


# ==============================================================================
# Step 2: PBC repair and trajectory conversion
# ==============================================================================
def convert_trajectory(ref_pdb_path, traj_path, complex_atom_indices,
                       n_complex_atoms, lig_resname, outdir):
    """Load XTC, fix PBC, strip ions/water, save as Amber NetCDF."""
    import mdtraj as md

    print()
    print("[Step 2/4] Converting trajectory...")

    print(f"  Loading: {traj_path}")
    traj = md.load(traj_path, top=ref_pdb_path)
    print(f"  Loaded: {traj.n_frames} frames, {traj.n_atoms} atoms")

    if traj.unitcell_lengths is not None:
        print("  Repairing PBC artifacts...")
        traj = _fix_pbc(traj, lig_resname)
    else:
        print("  No periodic box info, skipping PBC repair")

    traj_complex = traj.atom_slice(complex_atom_indices)
    print(f"  Complex trajectory: {traj_complex.n_frames} frames, "
          f"{traj_complex.n_atoms} atoms")

    assert traj_complex.n_atoms == n_complex_atoms, \
        (f"Atom count mismatch: traj={traj_complex.n_atoms} vs "
         f"prmtop={n_complex_atoms}")

    nc_path = os.path.join(outdir, 'complex.nc')
    _write_amber_netcdf(traj_complex, nc_path)
    print(f"  Wrote: {nc_path}")

    return nc_path


def _fix_pbc(traj, lig_resname):
    """Robust PBC repair."""
    import mdtraj as md

    top = traj.topology
    protein_idx = top.select('protein')
    lig_idx = top.select(f'resname {lig_resname}')

    if len(protein_idx) == 0:
        print("    WARNING: No protein atoms found, skipping PBC repair")
        return traj

    # Make molecules whole
    try:
        traj = traj.image_molecules(inplace=False, make_whole=True)
        print("    ✓ image_molecules completed")
    except Exception as e:
        print(f"    image_molecules failed ({e}), using manual method...")
        molecules = list(traj.topology.find_molecules())
        for fi in range(traj.n_frames):
            box = traj.unitcell_lengths[fi]
            for mol in molecules:
                mol_idx = [atom.index for atom in mol]
                if len(mol_idx) < 2:
                    continue
                coords = traj.xyz[fi]
                ref = coords[mol_idx[0]].copy()
                for i in range(1, len(mol_idx)):
                    diff = coords[mol_idx[i]] - ref
                    coords[mol_idx[i]] -= box * np.round(diff / box)
        print("    ✓ Manual make-whole completed")

    # Continuous-frame tracking
    ref_com = np.mean(traj.xyz[0, protein_idx, :], axis=0)
    for fi in range(1, traj.n_frames):
        box = traj.unitcell_lengths[fi]
        com = np.mean(traj.xyz[fi, protein_idx, :], axis=0)
        shift = box * np.round((com - ref_com) / box)
        if np.any(np.abs(shift) > 0.01):
            traj.xyz[fi] -= shift
        ref_com = np.mean(traj.xyz[fi, protein_idx, :], axis=0)
    print("    ✓ Continuous-frame tracking completed")

    # Center protein
    for fi in range(traj.n_frames):
        com = np.mean(traj.xyz[fi, protein_idx, :], axis=0)
        traj.xyz[fi] += traj.unitcell_lengths[fi] / 2.0 - com
    print("    ✓ Protein centered")

    # Image ligand
    if len(lig_idx) > 0:
        for fi in range(traj.n_frames):
            pcom = np.mean(traj.xyz[fi, protein_idx, :], axis=0)
            lcom = np.mean(traj.xyz[fi, lig_idx, :], axis=0)
            box = traj.unitcell_lengths[fi]
            shift = box * np.round((lcom - pcom) / box)
            if np.any(np.abs(shift) > 0.01):
                traj.xyz[fi, lig_idx, :] -= shift
        print("    ✓ Ligand imaging completed")

    # Validation
    bb_idx = top.select('protein and backbone')
    if len(bb_idx) > 0:
        bb_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=bb_idx) * 10
        print(f"    Backbone RMSD: mean={np.mean(bb_rmsd):.2f} Å, "
              f"max={np.max(bb_rmsd):.2f} Å")
        if np.max(bb_rmsd) > 50.0:
            print("    ⚠ WARNING: High backbone RMSD")

    if len(lig_idx) > 0:
        lig_rmsd = md.rmsd(traj, traj, frame=0, atom_indices=lig_idx) * 10
        print(f"    Ligand RMSD: mean={np.mean(lig_rmsd):.2f} Å, "
              f"max={np.max(lig_rmsd):.2f} Å")

    return traj


def _write_amber_netcdf(traj, nc_path):
    """Write trajectory as Amber NetCDF format."""
    import netCDF4

    n_frames = traj.n_frames
    n_atoms = traj.n_atoms

    ncfile = netCDF4.Dataset(nc_path, 'w', format='NETCDF3_64BIT')
    ncfile.Conventions = 'AMBER'
    ncfile.ConventionVersion = '1.0'
    ncfile.application = 'AMBER'
    ncfile.program = 'gbsa_new.py'

    ncfile.createDimension('frame', None)
    ncfile.createDimension('spatial', 3)
    ncfile.createDimension('atom', n_atoms)

    has_box = traj.unitcell_lengths is not None
    if has_box:
        ncfile.createDimension('cell_spatial', 3)
        ncfile.createDimension('cell_angular', 3)
        ncfile.createDimension('label', 5)

    spatial = ncfile.createVariable('spatial', 'c', ('spatial',))
    spatial[:] = np.array(['x', 'y', 'z'])

    time_var = ncfile.createVariable('time', 'f', ('frame',))
    time_var.units = 'picosecond'
    if traj.time is not None and len(traj.time) == n_frames:
        time_var[:] = traj.time.astype(np.float32)
    else:
        time_var[:] = np.arange(n_frames, dtype=np.float32)

    coords_var = ncfile.createVariable('coordinates', 'f',
                                        ('frame', 'atom', 'spatial'))
    coords_var.units = 'angstrom'
    coords_var[:] = traj.xyz * 10.0

    if has_box:
        cs = ncfile.createVariable('cell_spatial', 'c', ('cell_spatial',))
        cs[:] = np.array(['a', 'b', 'c'])

        ca = ncfile.createVariable('cell_angular', 'c',
                                    ('cell_angular', 'label'))
        ca[0] = np.array(list('alpha'))
        ca[1] = np.array(list('beta '))
        ca[2] = np.array(list('gamma'))

        cl = ncfile.createVariable('cell_lengths', 'd',
                                    ('frame', 'cell_spatial'))
        cl.units = 'angstrom'
        cl[:] = traj.unitcell_lengths * 10.0

        cang = ncfile.createVariable('cell_angles', 'd',
                                      ('frame', 'cell_angular'))
        cang.units = 'degree'
        cang[:] = traj.unitcell_angles

    ncfile.close()


# ==============================================================================
# Step 3: Validate consistency
# ==============================================================================
def validate(complex_prmtop, receptor_prmtop, ligand_prmtop, nc_path):
    """Check atom counts are consistent."""
    import parmed
    import netCDF4

    print()
    print("[Step 3/4] Validating consistency...")

    cp = parmed.load_file(complex_prmtop)
    rp = parmed.load_file(receptor_prmtop)
    lp = parmed.load_file(ligand_prmtop)

    nc = netCDF4.Dataset(nc_path, 'r')
    nc_atoms = nc.dimensions['atom'].size
    nc_frames = nc.dimensions['frame'].size
    nc.close()

    print(f"  complex.prmtop:  {len(cp.atoms)} atoms")
    print(f"  receptor.prmtop: {len(rp.atoms)} atoms")
    print(f"  ligand.prmtop:   {len(lp.atoms)} atoms")
    print(f"  complex.nc:      {nc_atoms} atoms, {nc_frames} frames")

    ok = True
    if nc_atoms != len(cp.atoms):
        print(f"  ✗ traj({nc_atoms}) != complex({len(cp.atoms)})")
        ok = False
    if len(rp.atoms) + len(lp.atoms) != len(cp.atoms):
        print(f"  ✗ rec({len(rp.atoms)}) + lig({len(lp.atoms)}) "
              f"!= complex({len(cp.atoms)})")
        ok = False

    if ok:
        print("  ✓ All atom counts consistent")
    else:
        print("  ERROR: Atom count mismatch!")
        sys.exit(1)

    return nc_frames


# ==============================================================================
# Step 4: Run MMPBSA.py
# ==============================================================================
def run_mmpbsa(complex_prmtop, receptor_prmtop, ligand_prmtop, nc_path,
               args, startframe, endframe, outdir):
    """Write mmpbsa.in and run MMPBSA.py."""
    print()
    print("[Step 4/4] Running MMPBSA.py...")

    method = args.method
    interval = args.interval
    decomp = args.decomp

    mmpbsa_in = os.path.join(outdir, 'mmpbsa.in')
    endframe_str = f"endframe={endframe}," if endframe > 0 else ""

    decomp_section = ""
    if decomp:
        decomp_section = """
&decomp
  idecomp=1, csv_format=1,
  dec_verbose=0,
/
"""

    method_sections = ""
    if method in ('gb', 'both'):
        method_sections += f"""&gb
  igb={args.igb}, saltcon={args.saltcon:.3f},
/
"""
    if method in ('pb', 'both'):
        method_sections += f"""&pb
  istrng={args.istrng:.3f}, fillratio={args.fillratio:.1f},
  inp={args.inp}, radiopt={args.radiopt},
  cavity_surften={args.cavity_surften:.4f}, cavity_offset={args.cavity_offset:.4f},
/
"""

    with open(mmpbsa_in, 'w') as f:
        f.write(f"""&general
  startframe={startframe}, {endframe_str}
  interval={interval},
  verbose=2, keep_files=0,
/
{method_sections}{decomp_section}""")

    method_label = {'gb': 'MM-GBSA', 'pb': 'MM-PBSA',
                    'both': 'MM-GBSA + MM-PBSA'}[method]
    print(f"  Method: {method_label}")
    if method in ('gb', 'both'):
        print(f"  GB: igb={args.igb}, saltcon={args.saltcon}")
    if method in ('pb', 'both'):
        print(f"  PB: istrng={args.istrng}, fillratio={args.fillratio}, "
              f"inp={args.inp}, radiopt={args.radiopt}")
    print(f"  Frames: start={startframe}, "
          f"end={'all' if endframe == 0 else endframe}, interval={interval}")

    cmd = [
        'MMPBSA.py', '-O',
        '-i', mmpbsa_in,
        '-cp', complex_prmtop,
        '-rp', receptor_prmtop,
        '-lp', ligand_prmtop,
        '-y', nc_path,
    ]

    print(f"  Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=outdir, capture_output=False)

    if result.returncode != 0:
        print(f"\n  ERROR: MMPBSA.py exit code {result.returncode}")
        sys.exit(1)

    # Collect MMPBSA.py output files into outdir
    # MMPBSA.py writes to cwd, which is outdir
    results_file = os.path.join(outdir, 'FINAL_RESULTS_MMPBSA.dat')
    if os.path.exists(results_file):
        print()
        print("=" * 60)
        print(f" {method_label} Results")
        print("=" * 60)
        with open(results_file) as f:
            print(f.read())
    else:
        print("  WARNING: FINAL_RESULTS_MMPBSA.dat not found")


# ==============================================================================
# Cleanup
# ==============================================================================
def cleanup_intermediate(outdir, keep):
    """Remove intermediate files unless --keep-intermediate is set."""
    if keep:
        return

    intermediate_patterns = [
        'complex.inpcrd', 'receptor.inpcrd', 'ligand.inpcrd',
        'reference.frc', 'mutant.frc',
    ]
    for fname in intermediate_patterns:
        path = os.path.join(outdir, fname)
        if os.path.exists(path):
            os.remove(path)

    # MMPBSA.py temp files
    for fname in os.listdir(outdir):
        if fname.startswith('_MMPBSA_') or fname == 'mmpbsa.in':
            path = os.path.join(outdir, fname)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


# ==============================================================================
# Main
# ==============================================================================
def main():
    args = parse_args()

    workdir = os.path.abspath(args.workdir)
    datadir = args.datadir or os.path.join(workdir, 'mlmm_output')
    outdir = args.outdir or os.path.join(workdir, 'gbsa_results')

    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    ref_pdb = os.path.join(datadir, 'equil_npt.pdb')
    traj_xtc = os.path.join(datadir, 'simulation.xtc')
    ligand_sdf = os.path.join(datadir, '../ligand.sdf')

    # Check inputs
    for fpath, desc in [(ref_pdb, 'Reference PDB'),
                         (traj_xtc, 'Trajectory'),
                         (ligand_sdf, 'Ligand SDF')]:
        if not os.path.exists(fpath):
            print(f"ERROR: {desc} not found: {fpath}")
            sys.exit(1)

    method_label = {'gb': 'MM-GBSA', 'pb': 'MM-PBSA',
                    'both': 'MM-GBSA + MM-PBSA'}[args.method]

    print("=" * 60)
    print(f" {method_label} Pipeline (OpenMM → Amber → MMPBSA.py)")
    print("=" * 60)
    print(f"  Working dir:  {workdir}")
    print(f"  Data dir:     {datadir}")
    print(f"  Output dir:   {outdir}")
    print(f"  Ref PDB:      {ref_pdb}")
    print(f"  Trajectory:   {traj_xtc}")
    print(f"  Ligand SDF:   {ligand_sdf}")
    print(f"  Lig resname:  {args.ligand_resname}")
    print(f"  Method:       {method_label}")
    print(f"  Ligand FF:    {args.ligand_ff}")
    print(f"  Protein FF:   {args.forcefields + args.extra_forcefields}")
    print()

    # Step 1: Create prmtop files (output to outdir)
    (complex_prmtop, receptor_prmtop, ligand_prmtop,
     n_complex, complex_indices) = create_prmtop_files(
        ref_pdb, ligand_sdf, args.ligand_resname,
        args.forcefields, args.extra_forcefields,
        args.ligand_ff, outdir, igb=args.igb,
    )

    # Step 2: Convert trajectory (output to outdir)
    nc_path = convert_trajectory(
        ref_pdb, traj_xtc, complex_indices,
        n_complex, args.ligand_resname, outdir,
    )

    # Step 3: Validate
    n_frames = validate(complex_prmtop, receptor_prmtop, ligand_prmtop, nc_path)

    # Auto-adjust endframe
    endframe = args.endframe
    if endframe == 0 or endframe > n_frames:
        endframe = n_frames

    # Step 4: Run MMPBSA.py (cwd = outdir)
    run_mmpbsa(
        complex_prmtop, receptor_prmtop, ligand_prmtop, nc_path,
        args, args.startframe, endframe, outdir,
    )

    # Cleanup
    cleanup_intermediate(outdir, args.keep_intermediate)

    print()
    print(f"Done! Results in: {outdir}")


if __name__ == '__main__':
    main()
