# AutoCG

AutoCG generates well-aligned reactant/product structures for transition state (TS) searches from reaction SMILES (with optional atom mapping) or 3D geometries.

## Requirements

- Python ≥ 3.10
- numpy, scipy ≥ 1.11
- cclib ≥ 1.7.1
- ORCA or Gaussian available on PATH (select with `-ca orca|gaussian`)
- Optional: CREST 6.2.3 for TS conformer sampling (https://github.com/crest-lab/crest)

## Installation

```bash
conda create -n autocg python=3.10
conda activate autocg
git clone https://github.com/Romarin87/autoCG
cd autoCG
pip install .
# or editable
pip install -e .
```

## Quick start

- From mapped SMILES:
  ```bash
  autocg "[H:1][H:2].[C-:3]#[O+:4]>>[O:4]=[C:3]([H:1])[H:2]" \
    -sd /path/to/save -wd /path/to/work
  ```
- From unmapped SMILES (AutoCG will infer bond changes):
  ```bash
  autocg "[H][H].[C-]#[O+]>>C=O" -sd /path/to/save -wd /path/to/work
  ```
- From geometry + bond changes (`input.com`):
  ```bash
  autocg "/path/to/input.com" -sd /path/to/save -wd /path/to/work
  ```
  Minimal `input.com` example:
  ```
  0 1  # charge multiplicity
  H 0.35400009037695684 0.0 0.0
  H -0.35400009037695684 0.0 0.0
  C 0.5640604828962581 0.0 4.0
  O -0.5640604828962581 0.0 4.0
  # Must give empty line
  1 2 B  # Bond change: B -> break
  1 3 F  # Bond change: F -> form
  2 3 F
  ```

Important options:
- `-nc` (#conformers), `-ts/-fs/-bs` (bond scaling), `-p` (UFF preopt), `-se` (stereo enumeration), `-uc` (CREST), `-ca` (calculator).
- `-cc 0` disables connectivity checks (useful for special bridged rings/strained structures where long bonds may be flagged as disconnected).
- `-cs 0` disables stereochemical checks.

Calculator setup: ensure `g09/g16` or `orca` is on `PATH`; pick with `-ca gaussian|orca`. CREST is optional but needed for `-uc 1`.

## Inputs

- **SMILES**:
  - If atom mapping is present, it is used directly to infer bond changes; otherwise AutoCG infers minimal bond form/break automatically.
  - Quote the SMILES in the shell to avoid expansion.
  - Example (mapped): `[C:1](=[O:2])[O:3][H:4]>>[C:1](=[O:2])[O-:3].[H+:4]`
  - Example (unmapped): `[C](=O)O>>CO`
- **Geometry**: `input.com` style must include charge/multiplicity on the first line, coordinates, a blank line, then bond-change lines (`B` for break, `F` for form).
- Charge/multiplicity should be provided; missing values are inferred when possible but may be ambiguous.

## Outputs

Each generated conformer is saved under `result_<idx>/` in the save directory:
- `initial_ts.xyz`: starting pseudo-TS geometry
- `TS_to_R.xyz`, `TS_to_P.xyz`: constrained scans toward reactant/product
- `UFF_R.xyz`, `UFF_P.xyz` (if preopt): UFF preoptimization trajectories
- `opt_R.xyz`, `opt_P.xyz`: QC relaxation trajectories
- `R.xyz`, `P.xyz` (+ `.com`): final relaxed reactant/product structures
- `guess.log`: summary of generation and matching results

## Notes

- If reaction SMILES carries atom mapping, AutoCG uses it directly to infer bond changes; otherwise, bond changes are inferred automatically.
- Geometry input must include charge/multiplicity and bond-change info; initial alignment is not required.
- Spectator molecules that do not participate in bond breaking/formation are ignored and not modeled.
- Algorithm details on bond-change inference and generation follow Chem. Sci. 2018 / J. Chem. Inf. Model. 2012 (see: [https://pubs.acs.org/doi/full/10.1021/ci3002217](https://pubs.acs.org/doi/full/10.1021/ci3002217)).
- Original extended instructions and calculator implementation notes are kept in `subpage/details.md`.

## License

BSD 3-Clause License.
