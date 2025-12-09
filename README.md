# AutoCG

AutoCG generates well-aligned reactant/product structures for transition state (TS) searches from reaction SMILES (with optional atom mapping) or 3D geometries.

## Requirements

- Python ≥ 3.8
- numpy, scipy ≥ 1.11
- cclib ≥ 1.7.1
- ORCA or Gaussian available on PATH (select with `-ca orca|gaussian`)
- Optional: CREST 6.2.3 for TS conformer sampling

## Installation

```bash
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
- From geometry + bond changes (`input.com`):
  ```bash
  autocg "/path/to/input.com" -sd /path/to/save -wd /path/to/work
  ```

Important options: `-nc` (#conformers), `-ts/-fs/-bs` (bond scaling), `-p` (UFF preopt), `-se` (stereo enumeration), `-uc` (CREST), `-ca` (calculator).

Calculator setup: ensure `g09/g16` or `orca` is on `PATH`; pick with `-ca gaussian|orca`. CREST is optional but needed for `-uc 1`.

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
- Algorithm details on bond-change inference and generation follow Chem. Sci. 2018 / J. Chem. Inf. Model. 2012 (see: [https://pubs.acs.org/doi/full/10.1021/ci3002217](https://pubs.acs.org/doi/full/10.1021/ci3002217)).
- Original extended instructions and calculator implementation notes are kept in `subpage/details.md`.

## License

BSD 3-Clause License.

## Contact

[kyunghoonlee@kaist.ac.kr](mailto:kyunghoonlee@kaist.ac.kr)
