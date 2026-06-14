# OpenMM MM/ML-MM + MMPBSA.py Workflow

This repository provides a small, self-contained example workflow for running pure MM or ML/MM molecular dynamics simulations in OpenMM, followed by endpoint binding free energy calculation with `MMPBSA.py`.

The workflow supports:

- Pure MM simulations using OpenMM and OpenFF
- ML/MM simulations using OpenMM-ML
- Conversion of OpenMM outputs to Amber-compatible formats
- Endpoint binding free energy estimation with `MMPBSA.py`

## Repository Contents

| File | Description |
|---|---|
| `environment_*.yml` | Conda environment specification for OpenMM, OpenFF, OpenMM-ML, and AmberTools |
| `mm.py` | Pure MM simulation script using OpenMM and OpenFF |
| `mlmm.py` | ML/MM simulation script using OpenMM, OpenFF, and OpenMM-ML |
| `gbsa.py` | Converts OpenMM outputs to Amber-compatible files and runs `MMPBSA.py` |
| `gbsa.sh` | Convenience wrapper for `gbsa.py` |
| `protein.pdb` | Example protein input file |
| `ligand.sdf` | Example ligand input file |

## Quick Start

### 1. Create the Conda environment

```bash
conda env create -f environment_aceff_aimnet.yml
conda activate mlmm_gbsa_aceff_aimnet
```

### 2. Prepare input files

Place your input files in the working directory:

```text
protein.pdb
ligand.sdf
```

Before running the simulations, check the force field settings in `mm.py` and `mlmm.py`.

### 3. Run an MM or ML/MM simulation

For a pure MM simulation:

```bash
python mm.py
```

For an ML/MM simulation:

```bash
python mlmm.py
```

Simulation outputs will be written to:

```text
md_output/
```

The main output files are:

```text
md_output/equil_npt.pdb
md_output/simulation.xtc
```

### 4. Run MMPBSA.py

```bash
bash gbsa.sh
```

Binding free energy results will be written to:

```text
gbsa_results/
```

The main result file is:

```text
gbsa_results/FINAL_RESULTS_MMPBSA.dat
```

## Notes

- For `mlmm.py` when using aceff or aimnet2 model, remember to modify the charge and multiplicity according to your ligand.
- `gbsa.sh` passes `--datadir ./md_output` to `gbsa.py`. Modify this path if your simulation outputs are stored elsewhere.
- If the ligand residue name in the PDB file is not `UNK`, set the correct value using `--ligand-resname`.
- Make sure the `--ligand-ff` option in `gbsa.sh` matches the small-molecule force field used in the MD simulation.
- Use `--method gb`, `--method pb`, or `--method both` in `gbsa.sh` to control the solvent model used by `MMPBSA.py`.

## Example Workflow

```bash
conda env create -f environment_aceff_aimnet.yml
conda activate mlmm_gbsa_aceff_aimnet

python mlmm.py
bash gbsa.sh
```
