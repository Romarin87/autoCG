"""
Run AutoCG on mapped reaction SMILES from a pickle and augment records with geometries and stereochemical SMILES.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
from typing import Dict, Tuple

import numpy as np
from ase import io as ase_io
from rdkit import Chem
from rdkit.Chem import SmilesParserParams
from rdkit.Geometry import Point3D


def _read_xyz(path: str) -> np.ndarray:
    atoms = ase_io.read(path)
    coords = atoms.get_positions()
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Unexpected coordinate shape from {path}: {coords.shape}")
    return np.asarray(coords, dtype=float)


def _smiles_with_coords(smiles: str, coords: np.ndarray) -> Tuple[str, str]:
    params = SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params=params)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")

    coords = np.asarray(coords, dtype=float)
    num_atoms = mol.GetNumAtoms()
    if coords.shape[0] != num_atoms or coords.shape[1] != 3:
        raise ValueError(f"Coordinate shape mismatch: expected ({num_atoms}, 3), got {coords.shape}")

    map_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(m == 0 for m in map_nums):
        raise ValueError("All atoms must carry atom map numbers to align coordinates.")
    if len(set(map_nums)) != len(map_nums):
        raise ValueError("Duplicate atom map numbers found in SMILES.")

    sorted_map_nums = sorted(map_nums)
    if len(sorted_map_nums) != coords.shape[0]:
        raise ValueError("Number of atom map numbers does not match coordinate count.")
    map_to_coord = {mn: coords[idx] for idx, mn in enumerate(sorted_map_nums)}

    conf = Chem.Conformer(num_atoms)
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        pos = map_to_coord[map_num]
        conf.SetAtomPosition(atom.GetIdx(), Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf, assignId=True)

    # Chem.AssignAtomChiralTagsFromStructure(mol, replaceExistingTags=True)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    # Chem.rdmolops.AssignStereochemistry(mol, force=True, cleanIt=True)

    mapped = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    mol_plain = Chem.RemoveHs(mol)
    for atom in mol_plain.GetAtoms():
        atom.SetAtomMapNum(0)
    plain = Chem.MolToSmiles(mol_plain, canonical=True, isomericSmiles=True)

    return mapped, plain


def _run_autocg(smiles: str, save_dir: str, work_dir: str, extra_args: Tuple[str, ...]) -> None:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    cmd = ["autocg", smiles, "-sd", save_dir, "-wd", work_dir, *extra_args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"autocg failed for {smiles}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _process_entry(
    entry: Dict, idx: int, save_root: str, work_root: str, extra_args: Tuple[str, ...]
) -> Dict:
    mapped = entry.get("reaction_smiles_mapped")
    if not mapped:
        raise ValueError(f"Entry {idx} missing reaction_smiles_mapped.")

    save_dir = os.path.abspath(os.path.join(save_root, f"job_{idx}"))
    work_dir = os.path.abspath(os.path.join(work_root, f"job_{idx}"))

    _run_autocg(mapped, save_dir, work_dir, extra_args)

    result_dir = os.path.join(save_dir, "result_1")
    r_xyz = os.path.join(result_dir, "R.xyz")
    p_xyz = os.path.join(result_dir, "P.xyz")
    react_geo = _read_xyz(r_xyz)
    prod_geo = _read_xyz(p_xyz)

    entry["reactant"]["geo"] = react_geo
    entry["product"]["geo"] = prod_geo

    try:
        r_smiles, p_smiles = mapped.split(">>")
    except ValueError as exc:
        raise ValueError(f"reaction_smiles_mapped must contain a single '>>': {mapped}") from exc

    r_mapped, r_plain = _smiles_with_coords(r_smiles, react_geo)
    p_mapped, p_plain = _smiles_with_coords(p_smiles, prod_geo)

    entry["reaction_smiles_mapped_stereo"] = f"{r_mapped}>>{p_mapped}"
    entry["reaction_smiles_stereo"] = f"{r_plain}>>{p_plain}"

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Run AutoCG over a pickle of reactions and attach geometries/stereochemical SMILES."
    )
    parser.add_argument("--input", "-i", required=True, help="Input pickle path.")
    parser.add_argument("--output", "-o", required=True, help="Output pickle path.")
    parser.add_argument(
        "--save-root",
        default="autocg_runs",
        help="Root directory for AutoCG -sd outputs (unique subfolders are created per entry).",
    )
    parser.add_argument(
        "--work-root",
        default="autocg_work",
        help="Root directory for AutoCG -wd working directories (unique subfolders are created per entry).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="If set, process only the first N entries from the input pickle.",
    )
    args, passthrough = parser.parse_known_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError("Input pickle must contain a list of reaction records.")
    if args.head is not None:
        data = data[: args.head]

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.work_root, exist_ok=True)

    updated = []
    for idx, entry in enumerate(data):
        try:
            updated_entry = _process_entry(
                entry, idx, args.save_root, args.work_root, tuple(passthrough)
            )
            updated.append(updated_entry)
        except Exception as exc:
            print(f"[WARN] Entry {idx} failed: {exc}")
            updated.append(entry)

    with open(args.output, "wb") as f:
        pickle.dump(updated, f)


if __name__ == "__main__":
    main()
