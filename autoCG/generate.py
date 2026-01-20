### python provided modules ###
import argparse
import concurrent.futures
import csv
import datetime
import os
import pickle
import subprocess
import sys
import time
from copy import deepcopy
from itertools import combinations
from pathlib import Path

### extra common libraries ###
import numpy as np
from scipy import spatial
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

### ace-reaction libraries ###
from autoCG import chem
from autoCG.Calculator import gaussian, orca, rdFF
from autoCG.utils import arranger, ic, process, stereo, conformation


class GuessGenerator:
    def __init__(
        self,
        calculator=None,
        ts_scale=1.33,
        form_scale=1.1,
        break_scale=1.8,
        protect_scale=0.95,
        max_separation=3.0,
        step_size=0.10,
        qc_step_size=0.2,
        num_relaxation=3,
    ):
        self.ts_scale = ts_scale
        self.form_scale = form_scale
        self.break_scale = break_scale
        self.protect_scale = protect_scale
        self.max_separation = max_separation
        self.step_size = step_size
        self.qc_step_size = qc_step_size
        self.num_step = 30  # May not change ...
        self.use_gurobi = True
        self.num_conformer = 1
        self.num_relaxation = num_relaxation
        self.scan_num_relaxation = 3
        self.scan_qc_step_size = 0.2
        self.unit_converter = 627.5095
        self.energy_criteria = 1.0 / self.unit_converter  # 1.0 kcal/mol
        self.library = "rdkit"
        self.save_directory = None
        self.working_directory = None
        self.uff_optimizer = rdFF.UFFOptimizer()
        self.preoptimize = True
        self.maximal_displacement = 0.2
        self.k = 5
        self.window = 0.5
        self.h_content = None
        self.num_process = 1
        # Check calculator
        if calculator is None:  # Default as Gaussian
            command = "g09"
            calculator = gaussian.Gaussian(command)  # Run with pm6
            # calculator = gaussian.FastGaussian(command,working_directory) # Run with pm6
            # command = 'orca'
            # calculator = orca.Orca(command,working_directory)
        else:
            if not os.path.exists(calculator.working_directory):
                calculator.change_working_directory(
                    os.getcwd()
                )  # Use the final working directory (may be save_directory) as default working directory, it's safe!
        self.calculator = calculator
        self.enumerate_stereo = False
        self.use_crest = False
        self.check_connectivity = True
        self.check_stereo = False

    def read_option(self, input_directory):
        if not os.path.exists(input_directory):
            print("No inputs are found!!!")
            return
        fd = open(os.path.join(input_directory, "option"), "r")
        for i in range(1000):
            line = fd.readline()
            if line == "":
                break
            line = line.split()
            if "scale" in line:
                idx = line.index("scale")
                try:
                    self.scale = float(line[idx + 2])
                except:
                    print("input scale is uncertain!!")
            if "b_scale" in line:
                idx = line.index("b_scale")
                try:
                    self.break_scale = float(line[idx + 2])
                except:
                    print("Cannot read b_scale value!!!")
            if "f_scale" in line:
                idx = line.index("f_scale")
                try:
                    self.form_scale = float(line[idx + 2])
                except:
                    print("Cannot read f_scale value!!!")
            if "ts_scale" in line:
                idx = line.index("ts_scale")
                try:
                    self.ts_scale = float(line[idx + 2])
                except:
                    print("Cannot read ts_scale uncertain!!")
            if "guess_step_size" in line:
                idx = line.index("guess_step_size")
                try:
                    self.step_size = float(line[idx + 2])
                except:
                    print("Cannot read step size for generarting guess!!")
            if "guess_num_conformer" in line:
                idx = line.index("guess_num_conformer")
                try:
                    self.num_conformer = int(line[idx + 2])
                except:
                    print("Cannot read the number of ts guess!!")
            if "guess_num_relaxation" in line:
                idx = line.index("guess_num_relaxation")
                try:
                    self.num_relaxation = int(line[idx + 2])
                except:
                    print(
                        "Cannot read the number of optimization for guess generation!!"
                    )
        fd.close()

    def read_qc_input(self, input_directory, file_name="qc_input"):
        qc_directory = os.path.join(input_directory, file_name)
        if os.path.exists(qc_directory):
            self.write_log(f"Read qc_input from {qc_directory}\n\n")
            content = ""
            with open(qc_directory) as f:
                for line in f:
                    content = content + line
            self.calcualtor.content = content
        else:
            self.write_log(
                f"Input template for quantum calculation is not found! Path will be found with default parameters!\n\n"
            )

    def change_save_directory(self, save_directory):
        if not os.path.exists(save_directory):
            os.system(f"mkdir {save_directory}")
            if not os.path.exists(save_directory):
                print("Directory not found! Cannot change save directory!")
            else:
                self.save_directory = save_directory
        else:
            self.save_directory = save_directory

    def change_working_directory(self, working_directory):
        if self.calculator is not None:
            self.calculator.change_working_directory(working_directory)
        else:
            print("Calculator does not exist!!! Define a calcualtor first!")

    def set_energy_criteria(self, criteria):
        self.energy_criteria = criteria / self.unit_converter

    def write_log(self, content, mode="a", file_name="guess"):
        if self.save_directory is None:
            # print ('No log directory!!! Need to give log directory!!!')
            return
        log_directory = os.path.join(self.save_directory, f"{file_name}.log")
        with open(log_directory, mode) as f:
            f.write(content)
            f.flush()

    def get_geometry(self, molecule):
        content = ""
        for atom in self._get_output_atoms(molecule):
            content = content + atom.get_content()
        return content

    def _get_output_atoms(self, molecule):
        atom_list = molecule.atom_list
        map_nums = []
        has_mapped = False
        for atom in atom_list:
            map_num = getattr(atom, "map_num", None)
            map_nums.append(map_num)
            if map_num is not None and map_num > 0:
                has_mapped = True
        if not has_mapped:
            return atom_list
        # Keep unmapped (0/None) atoms at the end; preserve original order on ties.
        order = list(range(len(atom_list)))
        order.sort(
            key=lambda i: (
                map_nums[i] is None or map_nums[i] <= 0,
                map_nums[i] if map_nums[i] is not None else 0,
                i,
            )
        )
        return [atom_list[i] for i in order]

    def write_geometry(self, molecule, name="test", extension="xyz"):
        if self.save_directory is None:
            return
        file_name = name + "." + extension
        content = self.get_geometry(molecule)
        with open(os.path.join(self.save_directory, file_name), "w") as f:
            if extension == "xyz":
                f.write(f"{len(molecule.atom_list)}\n\n")
            else:
                try:
                    f.write(f"{molecule.get_chg()} {molecule.get_multiplicity()}\n")
                except:
                    f.write("0 1\n")
            f.write(content + "\n")
            f.flush()

    def write_optimization(self, molecule, file_name=None, mode="a"):
        if self.save_directory is None:
            return
        if file_name is None:
            return
        os.makedirs(self.save_directory, exist_ok=True)
        content = self.get_geometry(molecule)
        e = molecule.get_energy()
        with open(os.path.join(self.save_directory, file_name), mode) as f:
            f.write(f"{len(molecule.atom_list)}\n")
            f.write(f"{e}\n")
            f.write(content)
            f.flush()

    def save_geometries(self, molecules, name="test", extension="xyz"):
        for i, molecule in enumerate(molecules):
            self.save_geometry(molecule, i, name)

    def save_result(self, reaction_info, mapping_info, matching_results):
        if self.save_directory is None:
            return
        with open(os.path.join(self.save_directory, "info.pkl"), "wb") as f:
            pickle.dump(reaction_info, f)
            pickle.dump(mapping_info, f)
            pickle.dump(matching_results, f)

    def get_reaction_info(self, reactant, product):
        reaction_info = dict()
        if self.use_gurobi:
            broken_bonds, formed_bonds = arranger.get_bond_change(reactant, product)
            reaction_info["b"] = broken_bonds
            reaction_info["f"] = formed_bonds
            return reaction_info
        else:  # Can be implemented later ...
            return None

    def get_reaction_info_from_mapping(self, reactant_smiles, product_smiles):
        """Use atom-mapped SMILES to derive bond changes directly."""
        def extract_bonds_and_map(mol):
            if mol is None:
                return None, None, None
            atom_map_to_idx = {}
            bond_orders = {}
            for atom in mol.GetAtoms():
                m = atom.GetAtomMapNum()
                if m > 0:
                    atom_map_to_idx[m] = atom.GetIdx()
            for bond in mol.GetBonds():
                m1 = bond.GetBeginAtom().GetAtomMapNum()
                m2 = bond.GetEndAtom().GetAtomMapNum()
                if m1 == 0 or m2 == 0:
                    return None, None, None  # incomplete mapping; fall back later
                if m1 > m2:
                    m1, m2 = m2, m1
                bond_orders[(m1, m2)] = bond.GetBondTypeAsDouble()
            return set(bond_orders.keys()), atom_map_to_idx, bond_orders

        r_mol = Chem.MolFromSmiles(reactant_smiles, sanitize=False)
        p_mol = Chem.MolFromSmiles(product_smiles, sanitize=False)
        r_bonds, r_map, r_orders = extract_bonds_and_map(r_mol)
        p_bonds, p_map, p_orders = extract_bonds_and_map(p_mol)
        if r_bonds is None or p_bonds is None:
            return None

        all_bonds = r_bonds | p_bonds
        reaction_info = {"b": [], "f": []}
        for m1, m2 in all_bonds:
            r_order = r_orders.get((m1, m2), 0)
            p_order = p_orders.get((m1, m2), 0)
            # Only treat appearance/disappearance as form/break; ignore pure bond-order changes
            if r_order == 0 and p_order > 0:
                reaction_info["f"].append(tuple(sorted((r_map[m1], r_map[m2]))))
            elif p_order == 0 and r_order > 0:
                reaction_info["b"].append(tuple(sorted((r_map[m1], r_map[m2]))))
        return reaction_info
    
    def find_spectator_molecules(self, reactant, reaction_info):
        # Find participating molecules (Non-participating is removed ...)
        reactant_atom_indices_for_each_molecule = (
            reactant.get_atom_indices_for_each_molecule()
        )
        molecule_indices = []
        # print (reactant_atom_indices_for_each_molecule)
        for bond in reaction_info["f"] + reaction_info["b"]:
            for i, atom_indices in enumerate(reactant_atom_indices_for_each_molecule):
                if bond[0] in atom_indices or bond[1] in atom_indices:
                    if i not in molecule_indices:
                        molecule_indices.append(i)
        # print (molecule_indices)
        molecule_list = reactant.get_molecule_list()
        not_used_molecule_indices = list(
            set(range(len(reactant_atom_indices_for_each_molecule)))
            - set(molecule_indices)
        )
        spectating_molecules = [molecule_list[i] for i in not_used_molecule_indices]

        used_atom_indices = []
        for i in molecule_indices:
            used_atom_indices = (
                used_atom_indices + reactant_atom_indices_for_each_molecule[i]
            )
        used_atom_indices.sort()

        # Make mapping that reduces the total_atom_indices to used_atom_indices
        mapping_info = dict()
        for i, index in enumerate(used_atom_indices):
            mapping_info[index] = i
        for i in range(len(reaction_info["f"])):
            start, end = reaction_info["f"][i]
            reaction_info["f"][i] = (mapping_info[start], mapping_info[end])
        for i in range(len(reaction_info["b"])):
            start, end = reaction_info["b"][i]
            reaction_info["b"][i] = (mapping_info[start], mapping_info[end])

        return reaction_info, mapping_info, spectating_molecules

    def get_ts_like_molecules(self, reactant, reaction_info):
        # Get stereoinformation of reactant (if exists)
        (
            potential_atom_indices,
            potential_bonds,
        ) = stereo.find_potential_chirals_and_stereo_bonds(reactant)
        reactant_atom_stereo_infos = dict()
        reactant_bond_stereo_infos = dict()
        for atom_index in potential_atom_indices:
            atom_stereo = stereo.get_atom_stereo(reactant, atom_index)
            reactant_atom_stereo_infos[atom_index] = atom_stereo
        for bond in potential_bonds:
            bond_stereo = stereo.get_bond_stereo(reactant, bond)
            reactant_bond_stereo_infos[bond] = bond_stereo

        original_z_list = reactant.get_z_list()

        # Divide by fixing stereos and enumerating stereos
        fixing_atom_stereo_infos = dict()
        fixing_bond_stereo_infos = dict()
        enumerating_atom_indices = []
        enumerating_bonds = []

        self.write_log(f"Stereo atoms: {reactant_atom_stereo_infos}\n")
        self.write_log(f"Stereo bonds: {reactant_bond_stereo_infos}\n")

        # First set fixing atom stereos
        reaction_participating_indices = []
        for bond in reaction_info["f"]:
            start, end = bond
            if start not in reaction_participating_indices:
                reaction_participating_indices.append(start)
            if end not in reaction_participating_indices:
                reaction_participating_indices.append(end)
        
        # Reaction not participating bonds ...
        for atom_index in reactant_atom_stereo_infos:
            if atom_index not in reaction_participating_indices:
                fixing_atom_stereo_infos[atom_index] = reactant_atom_stereo_infos[
                    atom_index
                ]

        for bond in reactant_bond_stereo_infos:
            start, end = bond
            if (
                start not in reaction_participating_indices
                or end not in reaction_participating_indices
            ):
                fixing_bond_stereo_infos[bond] = reactant_bond_stereo_infos[bond]

        # Get reduced molecule and modify reaction_info with reduced indices
        z_list = np.copy(reactant.get_z_list())
        adj_matrix = np.copy(reactant.get_matrix("adj"))
        for bond in reaction_info["f"]:
            start, end = bond
            adj_matrix[start][end] += 1
            adj_matrix[end][start] += 1
        
        # Sample one conformer with valid molecule
        ts_molecule = chem.Molecule([z_list, adj_matrix, None, None])
        ts_molecule.chg = reactant.get_chg()
        for src_atom, dst_atom in zip(reactant.atom_list, ts_molecule.atom_list):
            dst_atom.map_num = getattr(src_atom, "map_num", None)
        ts_molecule = ts_molecule.get_valid_molecule()
        ts_chg_list = ts_molecule.get_chg_list()
        ts_bo = ts_molecule.get_bo_matrix()
        ts_molecules = ts_molecule.sample_conformers(1)
        for ts_molecule in ts_molecules:
            ts_molecule.bo_matrix = ts_bo
            ts_molecule.atom_feature['chg'] = ts_chg_list

        # Stereo sampling first, if activated ...
        if self.enumerate_stereo:
            self.write_log('Considering stereoisomer enumeration ...'+'\n')
            st = datetime.datetime.now()
            final_ts_molecules = []
            valency_list = np.sum(adj_matrix,axis=1)
            complex_indices = np.where(valency_list>4)[0].tolist()
            #print (complex_indices)
            for ts_molecule in ts_molecules:
                # Change stereo to reactant
                ts_molecule = stereo.get_desired_stereoisomer(
                    ts_molecule, fixing_atom_stereo_infos, fixing_bond_stereo_infos
                )
                # Enumerate stereoisomer
                enumerated_ts_molecules = stereo.enumerate_stereoisomers(
                    ts_molecule, reaction_participating_indices, enumerating_bonds,complex_indices
                )
                final_ts_molecules += enumerated_ts_molecules
            ts_molecules = final_ts_molecules
            et = datetime.datetime.now()
            content =f'Total stereoisomer enumeration time: {et-st}' 
            print (content)
            self.write_log(content+'\n')
        ts_molecules = self.get_no_repeating_molecules(ts_molecules)
       
        self.write_log(f'Total {len(ts_molecules)} molecules enumerated ... \n\n') 
        # Adjust to TS ...
        for i in range(len(ts_molecules)):
            self.adjust_to_ts(ts_molecules[i],reaction_info,original_z_list)

        # Perform CREST here ...
        if self.use_crest:
            self.write_log('Performing CREST for searching lowest TS conformer ...'+'\n')
            st = datetime.datetime.now()
            # Perform CREST and re generate RP pairs in CREST folder ...
            final_ts_molecules = []
            constraints = reaction_info['b'] + reaction_info['f']
            for ts_molecule in ts_molecules:
                #ts_molecule.print_coordinate_list()
                conformers = conformation.sample_from_crest(ts_molecule,constraints,self.working_directory,self.num_process)
                #print ('crest conformers',len(conformers))
                conformers = self.get_no_repeating_molecules(conformers)
                #for conformer in conformers[:self.num_conformer]:
                    #print ('1################################')
                    #print (conformer.energy)
                    #conformer.print_coordinate_list()
                
                final_ts_molecules += conformers[:self.num_conformer]
            self.write_log(f'Final ts molecules: {len(final_ts_molecules)}\n')
            #random.shuffle(final_ts_molecules)
            # Remove repeating molecules ... (Almost same energy ...)
            print ([ts_molecule.energy for ts_molecule in final_ts_molecules])
            final_ts_molecules = self.get_no_repeating_molecules(final_ts_molecules)
            # Sort by conformer energy ...
            #final_ts_molecules = sorted(final_ts_molecules, key=lambda ts_molecule:ts_molecule.energy)
            # Gather only stable TS conformers ...           
            ts_molecules = final_ts_molecules
        #else:
        #    # Smiply fixed optimization for ts_molecules
        #    constraints = ts_molecule.get_bond_list(False)
        #    for ts_molecule in ts_molecules:
        #        print ('bf')
        #        ts_molecule.print_coordinate_list()
        #        self.calculator.relax_geometry(ts_molecule,constraints,num_relaxation = 100)
        #        print ('af')
        #        ts_molecule.print_coordinate_list()
            et = datetime.datetime.now()
            content =f'Total CREST execution time: {et-st}' 
            print (content)
            self.write_log(content+'\n\n')

        print ('Final generated TS structures:',len(ts_molecules))
        return ts_molecules, reactant_atom_stereo_infos, reactant_bond_stereo_infos


    def adjust_to_ts(self,molecule,reaction_info,original_z_list):
        constraints = dict()
        ts_z_list = molecule.get_z_list()
        
        for i in range(len(original_z_list)):
            molecule.atom_list[i].set_atomic_number(int(original_z_list[i]))
      
        for bond in molecule.get_bond_list(False):
            start, end = bond
            r1 = molecule.atom_list[start].get_radius()
            r2 = molecule.atom_list[end].get_radius()            
            if bond in reaction_info['b'] + reaction_info['f']:
                constraints[bond] = (r1+r2) * self.ts_scale
            else:
                constraints[bond] = (r1+r2) * self.form_scale
        
        # Get back to ts z list ...
        for i in range(len(ts_z_list)):
            molecule.atom_list[i].set_atomic_number(int(ts_z_list[i]))

        self.scale_bonds(molecule,constraints)
        
        fixing_atoms = []

        self.uff_optimizer.optimize_geometry(molecule,constraints,k=2000,add_atom_constraint = False)
        # Really restore into original z list ...
        for i in range(len(original_z_list)):
            molecule.atom_list[i].set_atomic_number(int(original_z_list[i]))

    def get_no_repeating_molecules(self,molecules):
        final_molecules = []
        window = self.window/627.5095
        energy_list = []
        for molecule in molecules:
            energy = molecule.energy
            if energy is None:
                energy = self.calculator.get_energy(molecule)
                molecule.energy = energy
            # energy = self.calculator.get_distance_potential(molecule)
            put_in = True
            for energy_prime in energy_list:
                if abs(energy_prime - energy) < window:
                    put_in = False
                    break
            if put_in:
                energy_list.append(energy)
                final_molecules.append(molecule)
        return final_molecules

    def scale_bonds(self,molecule,constraints):
        step_size = self.step_size
        max_num_step = 0
        coordinate_list = np.array(molecule.get_coordinate_list())
        update_q = dict()
        displacement = 0
        for constraint in constraints:
            start = constraint[0]
            end = constraint[1]
            delta_d = constraints[constraint] - ic.get_distance(coordinate_list,start,end)
            update_q[constraint] = delta_d
            displacement += abs(delta_d)
            if abs(delta_d)/step_size > max_num_step - 1:
                max_num_step = int(abs(delta_d)/step_size) + 1
        #print (update_q)
        n = len(coordinate_list)
        for constraint in update_q:
            update_q[constraint] /= max_num_step
        displacement /= max_num_step
        # Now use internal coordinate
        for i in range(max_num_step):
            converged = ic.update_geometry(molecule,update_q)
            if not converged:
                print ('scaling update not converged!!!')
            # Relax geometry

    def get_distance_potential(self,molecule):
        z_list = molecule.get_z_list()
        coordinate_list = molecule.get_coordinate_list()
        distance_matrix = spatial.distance_matrix(coordinate_list,coordinate_list)
        distance_matrix = np.where(distance_matrix<5,distance_matrix,0) # Only consider near by 2 or 3
        n = len(z_list)
        p = 2
        inverse_distance_matrix = np.where(distance_matrix>0,distance_matrix**p,0)
        for i in range(n):
            inverse_distance_matrix[i,:] *= z_list[i]
            inverse_distance_matrix[:,i] *= z_list[i]
        return np.sum(inverse_distance_matrix)

    def move_to_minima(self, molecule, formed_bonds, broken_bonds, file_name=None):
        step_size = self.step_size
        coordinate_list = molecule.get_coordinate_list()
        n = len(molecule.atom_list)
        energy_list = [molecule.energy]
        trajectory = []
        # total_bonds = broken_bonds + formed_bonds # Must fix broken bonds
        total_bonds = molecule.get_bond_list(False)
        # print ('total bonds: ',total_bonds)
        check_increase = False
        undesired_bonds = []
        self.write_optimization(molecule, file_name, "w")
        status_q = dict()
        num_step = 0
        file_addition = ''
        if file_name is not None:
            file_addition += file_name[6]
        for bond in formed_bonds:
            start, end = bond
            d = molecule.get_distance_between_atoms(start, end)
            r1 = molecule.atom_list[start].get_radius()
            r2 = molecule.atom_list[end].get_radius()
            target_d = (r1 + r2) * self.form_scale
            if d - target_d < 0:
                status_q[bond] = 0.0
            else:
                status_q[bond] = d - target_d
        max_distance = -1.0
        for bond in broken_bonds:
            start, end = bond
            d = molecule.get_distance_between_atoms(start, end)
            r1 = molecule.atom_list[start].get_radius()
            r2 = molecule.atom_list[end].get_radius()
            target_d = (r1 + r2) * self.break_scale
            status_q[bond] = d
            if target_d > max_distance:
                max_distance = target_d

        for bond in broken_bonds:
            d = status_q[bond] 
            if max_distance < d:
                status_q[bond] = 0.0
            else:
                status_q[bond] = max_distance - d

        for bond in status_q:
            if num_step < int(status_q[bond] / step_size) + 1:
                num_step = int(status_q[bond] / step_size) + 1

        if num_step == 0:
            return
        # Determine num_step
        update_q = dict()
        for bond in total_bonds:
            if bond in formed_bonds:
                start, end = bond
                update_q[bond] = -status_q[bond] / num_step
            elif bond in broken_bonds:
                start, end = bond
                update_q[bond] = status_q[bond] / num_step
            else:
                update_q[bond] = 0.0
        for i in range(num_step):
            update = sum([abs(value) for value in list(update_q.values())])
            if update < 0.0001:  # If nothing to update ...
                print("No more coordinate to update ...")
                break
            # print (f'{i+1}th geometry ...')
            # molecule.print_coordinate_list()
            converged = ic.update_geometry(molecule, update_q)
            if not converged:
                break
            # For the case when constraints are less than possible total constraints
            if n * (n - 1) / 2 > len(update_q):
                self.calculator.relax_geometry_steep(
                    molecule,
                    update_q,
                    chg=None,
                    multiplicity=None,
                    file_name=f"scan_{i+1}_{file_addition}",
                    num_relaxation=self.scan_num_relaxation,
                    maximal_displacement=self.scan_qc_step_size,
                )
            self.write_optimization(molecule, file_name, "a")

    def uff_optimization(self, reactant, constraints=[],k=None, file_name=None):
        original_reactant = reactant.copy()  # Geometry exists ...
        # First optimize molecules ...
        self.write_optimization(reactant, file_name, "w")
        #AllChem.GetUFFBondStretchParams(original_mol, 1, 2)
        maximal_displacement = self.maximal_displacement
        original_coordinate = reactant.get_coordinate_list()
        if k is None:
            k = self.k
        #self.uff_optimizer.optimize_geometry(reactant, constraints, maximal_displacement, k) This also works well !!!
        self.uff_optimizer.optimize_geometry(reactant, [], maximal_displacement, k,True)
        if not process.check_geometry(reactant.get_coordinate_list()):
            print ('UFF optimization not sucessful!!!')
            print ('Loading original geometry ...')
            process.locate_molecule(reactant,original_coordinate)
        reactant.energy = self.calculator.get_energy(reactant)
        self.write_optimization(reactant, file_name, "a")

    def relax_geometry(self, reactant, constraints, file_name=None):
        energy_list = [self.calculator.get_energy(reactant)]
        self.write_optimization(reactant, file_name, "w")
        old_content = self.calculator.content
        if self.h_content is not None:
            self.calculator.content = self.h_content
        addition = ''
        if file_name is not None:
            addition += file_name[4]
        for i in range(self.num_step):
            self.calculator.relax_geometry_steep(
                reactant,
                constraints,
                None,
                None,
                f"relax_{i+1}_{addition}",
                self.num_relaxation,
                self.qc_step_size,
            )
            self.write_optimization(reactant, file_name, "a")
            energy_list.append(reactant.energy)
            if energy_list[-1] - energy_list[-2] > -self.energy_criteria:
                print(
                    f"E criteria MET: \u0394E = {energy_list[-1] - energy_list[-2]} hartree"
                )
                break
            # print ('After optimization ...')
            # reactant.print_coordinate_list()
        self.calculator.content = old_content
        # print ('#### Optimized geometry ####')
        # reactant.print_coordinate_list()

    def identify_connectivity(self,optimized_r, reactant, optimized_p, product):
        matching_result = [True, True]
        matching_result[0], coeff = process.is_same_connectivity(
            reactant, optimized_r
        )
        # print (reduced_reactant.get_smiles('ace'))
        if product is not None:
            matching_result[1], coeff = process.is_same_connectivity(
                product, optimized_p
            )
        return matching_result

    def check_stereochemistry(self,reactant,reactant_atom_stereo_infos, reactant_bond_stereo_infos):
        # Check stereo result
        stereo_result = [True, True]
        for atom_index in reactant_atom_stereo_infos:
            final_atom_stereo_info = stereo.get_atom_stereo(
                reactant, atom_index
            )
            if final_atom_stereo_info * reactant_atom_stereo_infos[atom_index] < 0:
                stereo_result[0] = False
                break
        for bond in reactant_bond_stereo_infos:
            final_bond_stereo_info = stereo.get_bond_stereo(
                reactant, bond
            )
            if final_bond_stereo_info * reactant_bond_stereo_infos[bond] < 0:
                stereo_result[1] = False
                break
        return stereo_result

    def generate_RP_pair(self,ts_molecule,reaction_info=None,chg=None,multiplicity=None,save_directory=None,working_directory=None):
        # Write guess log for each directory
        st = datetime.datetime.now()
        old_save_directory = self.save_directory
        

        # Log changed into the specific folder directory
        if save_directory is not None:
            os.makedirs(save_directory, exist_ok=True)
            self.save_directory = save_directory

        self.write_geometry(ts_molecule,'initial_ts','xyz')

        self.calculator.clean_scratch()
        sd = self.save_directory
        print ('Starting generation ...')
        if True: # Later change into try/except ...
            reactant_molecules = ts_molecule.copy()
            reactant_molecules.energy = ts_molecule.energy
            product_molecules = ts_molecule.copy()
            product_molecules.energy = ts_molecule.energy
            if chg is not None:
                reactant_molecules.chg = chg
                product_molecules.chg = chg
            if multiplicity is not None:
                reactant_molecules.multiplicity = multiplicity
                product_molecules.multiplicity = multiplicity
            print("Moving to R optimization ...")
            self.write_log("moving to R ...\n")
            formed_bonds = reaction_info["f"]
            broken_bonds = reaction_info["b"]
            self.move_to_minima(
                reactant_molecules, broken_bonds, formed_bonds, "TS_to_R.xyz"
            )  # New module for finding reactant complex
            # reactant_molecules.print_coordinate_list()

            # Modify connectivity
            adj_matrix = reactant_molecules.get_adj_matrix()
            for bond in formed_bonds:
                start, end = bond
                adj_matrix[start][end] -= 1
                adj_matrix[end][start] -= 1
            reactant_molecules.adj_matrix = adj_matrix
            r_chg_list, r_bo = process.get_chg_list_and_bo_matrix_from_adj_matrix(
                reactant_molecules, chg
            )
            reactant_molecules.bo_matrix = r_bo
            reactant_molecules.atom_feature["chg"] = r_chg_list
            # Preoptimize with uff
            if self.preoptimize:
                self.uff_optimization(reactant_molecules, formed_bonds, file_name="UFF_R.xyz")

            self.write_log("###### R optimization #####\n")
            print("R optimization started.")
            self.relax_geometry(reactant_molecules, [], "opt_R.xyz")
            # self.relax_geometry(reactant_molecules, formed_bonds, "opt_R.xyz")
            self.write_log("R optimization finished !!!\n")
            print("R optimization finished.")
            # reactant_molecules.print_coordinate_list()
            self.write_geometry(reactant_molecules, "R", "xyz")
            self.write_geometry(reactant_molecules, "R", "com")
            self.calculator.clean_scratch()
            self.write_log("moving to P ...\n")
            print("Moving to P optimization ...")
            self.move_to_minima(
                product_molecules, formed_bonds, broken_bonds, "TS_to_P.xyz"
            )
            adj_matrix = product_molecules.get_adj_matrix()
            for bond in broken_bonds:
                start, end = bond
                adj_matrix[start][end] -= 1
                adj_matrix[end][start] -= 1
            product_molecules.adj_matrix = adj_matrix
            p_chg_list, p_bo = process.get_chg_list_and_bo_matrix_from_adj_matrix(
                product_molecules, chg
            )
            
            product_molecules.bo_matrix = p_bo
            product_molecules.atom_feature["chg"] = p_chg_list
            if self.preoptimize:
                self.uff_optimization(product_molecules, broken_bonds, file_name="UFF_P.xyz")

            undesired_bonds = []
            # print ('P geometry ...')
            # product_molecules.print_coordinate_list()
            self.write_log("###### P optimization #####\n")
            print("P optimization started.")
            # self.separated_reactant_optimization(product_molecules,
            self.relax_geometry(product_molecules, [], "opt_P.xyz")
            # self.relax_geometry(product_molecules, broken_bonds, "opt_P.xyz")
            self.write_log("P optimization finished !!!\n")
            print("P optimization finished.")
            self.write_geometry(product_molecules, "P", "xyz")
            self.write_geometry(product_molecules, "P", "com")
            self.calculator.clean_scratch()

            reactant_molecules.adj_matrix = None
            reactant_molecules.bo_matrix = None
            product_molecules.adj_matrix = None
            product_molecules.bo_matrix = None
            self.save_directory = old_save_directory
            return reactant_molecules, product_molecules
        # Marking ...
        else:
            self.write_log(f"{i+1}th RP structure generation failed ...\n")
            print (f'Generation failed ... Skiped for {i+1}th trial')
            self.save_directory = old_save_directory
            return None, None
           

    def get_oriented_RPs(
        self,
        reactant,
        product,
        reaction_info=None,
        chg=None,
        multiplicity=None,
        save_directory=None,
        working_directory=None,
    ):
        # Track original working directory so we can restore it later
        old_working_directory = getattr(self, "working_directory", None)
        working_directory_changed = False
        # Default save directory when not provided
        if save_directory is None:
            base_dir = working_directory if working_directory is not None else os.getcwd()
            save_directory = os.path.abspath(base_dir)
        # Check save_directory
        if type(save_directory) is str:
            if not os.path.exists(save_directory):
                os.system(f"mkdir {save_directory}")
                if not os.path.exists(save_directory):
                    print("Path does not exist! Guess is generated without log file!")
                    save_directory = None
        if working_directory is None:
            working_directory = save_directory
        elif not os.path.exists(working_directory):
            os.makedirs(working_directory,exist_ok=True)
            if not os.path.exists(working_directory):
                working_directory = save_directory

        if working_directory is not None:
            if self.calculator.working_directory == os.getcwd():
                self.calculator.change_working_directory(working_directory)
            os.system(f'mkdir -p {working_directory}')
            self.working_directory = working_directory
            working_directory_changed = True

        self.save_directory = save_directory
        starttime = datetime.datetime.now()
        self.write_log("##### Guess generator info #####\n", "w")
        self.write_log("Parameters:\n")
        self.write_log(f"  library: {self.library}\n")
        self.write_log(f"  ts_scale: {self.ts_scale}\n")
        self.write_log(f"  form_scale: {self.form_scale}\n")
        self.write_log(f"  broken_scale: {self.break_scale}\n")
        self.write_log(f"  num_conformer: {self.num_conformer}\n")
        self.write_log(f"  E_criteria (Delta E): {self.energy_criteria * self.unit_converter} kcal/mol\n")
        self.write_log(f"  step_size: {self.step_size}\n")
        self.write_log(f"  qc_max_displacement: {self.qc_step_size}\n")
        self.write_log(f"  num_relaxation: {self.num_relaxation}\n")
        self.write_log(f"  calculator: {self.calculator.command}\n\n")

        self.write_log(f"Start time: {starttime}\n")
        self.write_log(f"Save directory: {self.save_directory}\n\n")

        if reaction_info is None:
            self.write_log("Finding reaction information with gurobi!\n")
            if reactant is None or product is None:
                print(
                    "Reaction information is not sufficient !!! Cannot make good R/P conformation !!!"
                )
                exit()
            reaction_info = self.get_reaction_info(reactant, product)
            if reaction_info is None:
                self.write_log(
                    "Internal error occured for finding reaction information! Guess generation terminated ..."
                )
                exit()
        else:
            self.write_log("Using the provided reaction!\n")

        # Check reaction_info
        bond_breaks = reaction_info['b']
        adj_matrix = reactant.get_adj_matrix()
        for bond in bond_breaks:
            s, e = bond
            if adj_matrix[s][e] == 0:
                self.write_log("Wrong reaction info is given !!! Check the input again !!!\n")
                exit()
        reactant_copy = reactant.copy()
        if chg is not None:
            reactant_copy.chg = chg
        if multiplicity is not None:
            reactant_copy.multiplicity = multiplicity
        # Need to write reaction information here!
        self.write_log(f"####### Final reaction information #######\n")

        self.write_log(f"Reaction info: {reaction_info}\n")
        try:
            self.write_log(
                f'reaction SMILES: {reactant.get_smiles("ace")}>>{product.get_smiles("ace")}\n'
            )
        except:
            self.write_log("reaction SMILES cannot be generated !!!\n")

        # self.write_log(f'reaction SMILES: {reactant.get_smiles("ace")}>>{product.get_smiles("ace")}\n')
        self.write_log("Reaction info (mapped):\n")
        self.write_log(f"  bond form: {reaction_info['f']}\n")
        self.write_log(f"  bond dissociation: {reaction_info['b']}\n\n")
        print(f"Reaction info: {reaction_info}")
        # Mapped reaction_info (reduced), to reconstruct original, use mapping_info!
        self.write_log("Start to get ts-like molecules.\n")
        st = datetime.datetime.now()
        (
            reaction_info,
            mapping_info,
            spectating_molecules,
        ) = self.find_spectator_molecules(reactant, reaction_info)
        if len(spectating_molecules) == 0:
            reduced_reactant = reactant.copy()
            if product is not None:
                reduced_product = product.copy()
            else:
                reduced_product = None
        else:
            reduced_reactant = process.get_reduced_intermediate(reactant, mapping_info)
            if product is not None:
                reduced_product = process.get_reduced_intermediate(
                    product, mapping_info
                )
            else:
                reduced_product = None

        # Check total charge of reduced_reactant
        (
            ts_molecules,
            reactant_atom_stereo_infos,
            reactant_bond_stereo_infos,
        ) = self.get_ts_like_molecules(reduced_reactant, reaction_info)
        et = datetime.datetime.now()
        self.write_log(f"[{et}] Generated {len(ts_molecules)} TS conformers (time: {et-st}).\n\n")
        print(f"Generated {len(ts_molecules)} TS conformers.")
        # Note that ts_molecules only contain atoms of molecules that are participating in the reaction
        # Relax ts_molecules with constraints
        energy_list = []
        if len(ts_molecules) == 0:
            self.write_log("No conformer generated !!!\n")
            return [], reaction_info, mapping_info, []
        
        molecule = ts_molecules[0]
        # Build constraints for ts optimization
        atom_list = ts_molecules[0].atom_list
        bond_list = molecule.get_bond_list(False)
        self.write_log("Set up done!!!\n")
        if chg is None:
            chg = molecule.chg
            if chg is None:
                chg = reduced_reactant.get_chg()
                if chg is None:
                    self.write_log("Charge is not given !!!\n")
        if multiplicity is None:
            multiplicity = (np.sum(reduced_reactant.get_z_list()) - chg) % 2 + 1
        reactant_copy = reduced_reactant.copy()
        if chg is not None:
            reactant_copy.chg = chg
        if multiplicity is not None:
            reactant_copy.multiplicity = multiplicity

        current_directory = os.getcwd()
        original_z_list = reactant.get_z_list()
        n = len(ts_molecules)
        self.write_log(f"Total TS conformers to process: {n}\n")
        reactant_product_pairs = []
        matching_results = []
        stereo_results = []
        for i in range(n):
            # Write guess log for each directory
            ts_molecule = ts_molecules[i].copy()
            #self.adjust_to_ts(ts_molecule,reaction_info,original_z_list)
            st = datetime.datetime.now()
            # Log changed into the specific folder directory
            if save_directory is not None:
                new_save_directory = os.path.join(save_directory, f"result_{i + 1}")
                os.system(f"mkdir -p {new_save_directory}")
            else:
                new_save_directory = None
            
            if True:
                st = datetime.datetime.now()
                reactant_molecules, product_molecules = self.generate_RP_pair(ts_molecule,reaction_info,chg,multiplicity,new_save_directory,working_directory)
                et = datetime.datetime.now()
                self.write_log(f"[{et}] Conformer {i+1}: generation time {et - st}\n")
                self.calculator.clean_scratch()
                matching_result = self.identify_connectivity(reactant_molecules, reduced_reactant, product_molecules, reduced_product)
                reactant_molecules.bo_matrix = reduced_reactant.bo_matrix
                stereo_result = self.check_stereochemistry(reactant_molecules,reactant_atom_stereo_infos, reactant_bond_stereo_infos)
            else:
                matching_result = [False, False]
                stereo_result = [False, False]
                self.save_directory = save_directory
                self.write_log(f"{i+1}th RP structure generation failed ...\n")
                reactant_molecules = None
                product_molecules = None

            reactant_product_pairs.append((reactant_molecules,product_molecules))            
            matching_results.append(matching_result)
            stereo_results.append(stereo_result)
            self.write_log(f"Conformer {i+1} matching: {matching_result}, stereo: {stereo_result}\n")
        
        self.write_log(f"All matching results: {matching_results}\n")
        self.write_log(f"All stereo results: {stereo_results}\n")
        # self.save_result(reaction_info, mapping_info, matching_results)
        stereo_matching_indices = []
        connectivity_matching_indices = []
        # Get indices that both are resulted in matched
        if self.check_connectivity:
            for i, matching_result in enumerate(matching_results):
                if not matching_result[0] or not matching_result[1]:
                    continue
                connectivity_matching_indices.append(i)
        else:
            connectivity_matching_indices = list(range(len(matching_results)))
        if self.check_stereo:
            for i, stereo_result in enumerate(stereo_results):
                stereo_result = stereo_results[i]
                if stereo_result[0] and stereo_result[1]:
                    stereo_matching_indices.append(i)
        else:
            stereo_matching_indices = list(range(len(matching_results)))
        indices = list(set(connectivity_matching_indices) & set(stereo_matching_indices))
        indices.sort()
        printing_indices = [str(i+1) for i in indices]

        content = ",".join(printing_indices)
        self.write_log(f"Good conformer indices: {content}\n")
        print(f"Good conformers: {printing_indices}")
        # All molecules here, are in ts connectivity matrix!!! To reset, need to remove bonds using reaction_info
        endtime = datetime.datetime.now()

        self.write_log(f"Total time: {endtime - starttime}\n")

        
        # Reset working and save directory
        self.save_directory = None
        if working_directory_changed:
            self.working_directory = old_working_directory
        return (
            reactant_product_pairs,
            reaction_info,
            mapping_info,
            matching_results,
        )  # Reduced indices for reaction_info

    def get_oriented_RPs_from_smiles(
        self,
        reaction_smiles,
        chg=None,
        multiplicity=None,
        save_directory=None,
        working_directory=None,
    ):
        smiles_list = reaction_smiles.split(">>")
        reactant_smiles = smiles_list[0]
        product_smiles = smiles_list[1]
        reaction_info = None
        if ":" in reaction_smiles:
            reaction_info = self.get_reaction_info_from_mapping(
                reactant_smiles, product_smiles
            )
            if reaction_info is None:
                self.write_log("Warning: Atom mapping incomplete; falling back to automatic bond-change inference.\n")
        # Reactant must have geometry ...
        # rd_reactant = Chem.MolFromSmiles(reactant_smiles)
        reactant = chem.Intermediate(reactant_smiles)
        product = chem.Intermediate(product_smiles)
        return self.get_oriented_RPs(
            reactant,
            product,
            reaction_info,
            chg=chg,
            multiplicity=multiplicity,
            save_directory=save_directory,
            working_directory=working_directory,
        )

    def get_oriented_RPs_from_geometry(
        self,
        input_directory,
        chg=None,
        multiplicity=None,
        save_directory=None,
        working_directory=None,
    ):
        # Input directory: name.com, coordinates
        reaction_info = dict()
        reaction_info["f"] = []
        reaction_info["b"] = []
        atom_list = []
        name = "R"
        with open(input_directory) as f:
            # Read chg, multiplicity
            # name = f.readline().strip()
            chg, multiplicity = f.readline().strip().split()
            chg = int(chg)
            multiplicity = int(multiplicity)
            # Read molecule geometry
            content = f.readline()
            while content != "\n":
                element, x, y, z = content.split()
                x = float(x)
                y = float(y)
                z = float(z)
                new_atom = chem.Atom(element)
                new_atom.x = x
                new_atom.y = y
                new_atom.z = z
                atom_list.append(new_atom)
                content = f.readline()
            # Read coordinates
            while True:
                line = f.readline()
                try:
                    coordinate_info = line.strip().split()
                    start = int(coordinate_info[0]) - 1
                    end = int(coordinate_info[1]) - 1
                    reaction_type = coordinate_info[2].lower()  # b or f
                    bond = (start, end)
                    if start > end:
                        bond = (end, start)
                    reaction_info[reaction_type].append(bond)
                except:
                    break
        if name == "R":
            reactant = chem.Intermediate()
            reactant.atom_list = atom_list
            reactant.adj_matrix = process.get_adj_matrix_from_distance(reactant, 1.2)
            reactant.chg = chg
            reactant.sanitize()
        return self.get_oriented_RPs(
            reactant,
            None,
            reaction_info,
            chg=chg,
            multiplicity=multiplicity,
            save_directory=save_directory,
            working_directory=working_directory,
        )


def normalize_reaction_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if len(line) >= 2 and line[0] == line[-1] and line[0] in {"'", '"'}:
        line = line[1:-1].strip()
    return line or None


def parse_reaction_smiles(reaction: str) -> tuple[str, str]:
    if ">>" in reaction:
        parts = reaction.split(">>")
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", ""
    parts = reaction.split(">")
    if len(parts) == 3:
        return parts[0], parts[2]
    return "", ""


def collect_autocg_rows(
    reaction_dir: Path,
    output_base: Path,
    reactant_smiles: str,
    product_smiles: str,
    csv_charge: str,
    csv_mult: str,
) -> tuple[list[dict[str, str]], list[Path]]:
    result_dirs = sorted([p for p in reaction_dir.glob("result_*") if p.is_dir()])
    targets = result_dirs if result_dirs else [reaction_dir]
    rows = []
    missing = []
    for target_dir in targets:
        react = target_dir / "R.xyz"
        prod = target_dir / "P.xyz"
        ts = target_dir / "initial_ts.xyz"
        if not (react.is_file() and prod.is_file() and ts.is_file()):
            missing.append(target_dir)
            continue
        rows.append(
            {
                "React": str(react.resolve()),
                "Prod": str(prod.resolve()),
                "TS": str(ts.resolve()),
                "Label": str(target_dir.relative_to(output_base)),
                "Charge": csv_charge,
                "Mult": csv_mult,
                "ReactSmiles": reactant_smiles,
                "ProdSmiles": product_smiles,
            }
        )
    return rows, missing


def build_generator(args: argparse.Namespace) -> GuessGenerator:
    if args.calculator == "gaussian":
        calculator = gaussian.Gaussian()
    else:
        calculator = orca.Orca()
    generator = GuessGenerator(
        calculator,
        args.ts_scale,
        args.form_scale,
        args.break_scale,
        args.protect_scale,
        args.max_separation,
        args.step_size,
        args.qc_step_size,
        args.num_relaxation,
    )
    generator.library = args.library
    generator.num_conformer = args.num_conformer
    generator.set_energy_criteria(args.energy_criteria)
    generator.maximal_displcement = args.maximal_displacement
    generator.k = args.k
    generator.scan_num_relaxation = args.scan_num_relaxation
    generator.scan_qc_step_size = args.scan_qc_step_size
    generator.window = args.window
    generator.num_process = args.num_process
    generator.use_crest = bool(args.use_crest)
    generator.enumerate_stereo = bool(args.stereo_enumerate)
    generator.check_connectivity = bool(args.check_connectivity)
    generator.check_stereo = bool(args.check_stereo)
    generator.preoptimize = bool(args.preoptimize)
    print(generator.preoptimize)
    return generator


def run_reaction(
    reaction_arg: str,
    args: argparse.Namespace,
    save_directory: str | None,
    working_directory: str | None,
) -> None:
    generator = build_generator(args)
    if ">>" in reaction_arg:
        generator.get_oriented_RPs_from_smiles(
            reaction_arg,
            save_directory=save_directory,
            working_directory=working_directory,
        )
        return

    input_directory = reaction_arg
    folder_directory = os.path.dirname(input_directory)
    if folder_directory is None:
        folder_directory = os.getcwd()
    folder_directory = os.path.abspath(folder_directory)
    if save_directory is None:
        save_directory = folder_directory

    if args.use_h_content == 1:
        try:
            h_file_directory = os.path.join(folder_directory, "h_content")
            h_content = ""
            with open(h_file_directory) as f:
                for line in f:
                    h_content = h_content + line
        except Exception:
            h_content = None
    else:
        h_content = None
    generator.h_content = h_content
    generator.get_oriented_RPs_from_geometry(
        input_directory,
        save_directory=save_directory,
        working_directory=working_directory,
    )


def run_reaction_task(
    reaction_arg: str,
    args_dict: dict,
    save_directory: str,
    working_directory: str,
) -> None:
    args = argparse.Namespace(**args_dict)
    run_reaction(reaction_arg, args, save_directory, working_directory)


def run_batch(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    base_save_dir = (
        Path(args.save_directory).expanduser().resolve()
        if args.save_directory is not None
        else Path.cwd() / "gly_results"
    )
    base_work_dir = (
        Path(args.working_directory).expanduser().resolve()
        if args.working_directory is not None
        else base_save_dir
    )
    base_save_dir.mkdir(parents=True, exist_ok=True)
    base_work_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv is not None
        else base_save_dir / "reaction_inputs.csv"
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    reactions: list[str] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            reaction = normalize_reaction_line(line)
            if reaction:
                reactions.append(reaction)
    if not reactions:
        print(f"No reactions found in {input_path}", file=sys.stderr)
        return 1

    total = len(reactions)
    start = args.start
    end = args.end if args.end is not None else total
    if start < 1 or end < 1 or start > end or end > total:
        print(f"Invalid range: start={start}, end={end}, total={total}", file=sys.stderr)
        return 1

    width = len(str(total))
    tasks = []
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "React",
                "Prod",
                "TS",
                "Label",
                "Charge",
                "Mult",
                "ReactSmiles",
                "ProdSmiles",
            ],
        )
        writer.writeheader()

        for idx in range(start, end + 1):
            reaction = reactions[idx - 1]
            reactant_smiles, product_smiles = parse_reaction_smiles(reaction)
            reaction_dir = base_save_dir / f"{args.prefix}_{idx:0{width}d}"
            working_dir = base_work_dir / f"{args.prefix}_{idx:0{width}d}"
            if reaction_dir.exists() and not args.overwrite:
                rows, missing = collect_autocg_rows(
                    reaction_dir,
                    base_save_dir,
                    reactant_smiles,
                    product_smiles,
                    args.csv_charge,
                    args.csv_mult,
                )
                for row in rows:
                    writer.writerow(row)
                handle.flush()
                for target_dir in missing:
                    print(f"[{idx}/{total}] missing files in {target_dir}", file=sys.stderr)
                continue

            reaction_dir.mkdir(parents=True, exist_ok=True)
            working_dir.mkdir(parents=True, exist_ok=True)
            (reaction_dir / "reaction.smi").write_text(reaction + "\n", encoding="utf-8")

            if args.dry_run:
                print(f"[{idx}/{total}] {reaction_dir}")
                continue

            tasks.append(
                (
                    idx,
                    reaction,
                    str(reaction_dir),
                    str(working_dir),
                    reactant_smiles,
                    product_smiles,
                )
            )

        if args.dry_run:
            return 0

        if args.jobs < 1:
            print("--jobs must be >= 1.", file=sys.stderr)
            return 1

        args_dict = vars(args).copy()
        failures: list[int] = []
        if args.jobs == 1:
            for idx, reaction, save_dir, work_dir, reactant_smiles, product_smiles in tasks:
                run_reaction(reaction, args, save_dir, work_dir)
                rows, missing = collect_autocg_rows(
                    Path(save_dir),
                    base_save_dir,
                    reactant_smiles,
                    product_smiles,
                    args.csv_charge,
                    args.csv_mult,
                )
                for row in rows:
                    writer.writerow(row)
                handle.flush()
                for target_dir in missing:
                    print(f"[{idx}/{total}] missing files in {target_dir}", file=sys.stderr)
            return 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_meta = {
                executor.submit(
                    run_reaction_task,
                    reaction,
                    args_dict,
                    save_dir,
                    work_dir,
                ): (idx, save_dir, reactant_smiles, product_smiles)
                for idx, reaction, save_dir, work_dir, reactant_smiles, product_smiles in tasks
            }
            for future in concurrent.futures.as_completed(future_to_meta):
                idx, save_dir, reactant_smiles, product_smiles = future_to_meta[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[{idx}/{total}] failed: {exc}", file=sys.stderr)
                    failures.append(idx)
                    continue
                rows, missing = collect_autocg_rows(
                    Path(save_dir),
                    base_save_dir,
                    reactant_smiles,
                    product_smiles,
                    args.csv_charge,
                    args.csv_mult,
                )
                for row in rows:
                    writer.writerow(row)
                handle.flush()
                for target_dir in missing:
                    print(f"[{idx}/{total}] missing files in {target_dir}", file=sys.stderr)

        if failures:
            failures.sort()
            print(f"Failed reactions: {failures}", file=sys.stderr)
            return 1
    return 0


def main(argv=None):
    sys.stdout = sys.__stdout__
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reaction",
        nargs="?",
        help="Reaction SMILES (with >>) or path to geometry/bond-change input file",
    )
    parser.add_argument(
        "--input-file",
        "-i",
        type=str,
        help="Path to a file containing reaction SMILES, one per line.",
        default=None,
    )
    parser.add_argument(
        "--save_directory",
        "-sd",
        type=str,
        help="save directory for precomplex",
        default=None,
    )
    parser.add_argument(
        "--working_directory",
        "-wd",
        type=str,
        help="working directory for QC",
        default=None,
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Summary CSV path for batch runs.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for batch runs.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="reaction",
        help="Folder name prefix for batch outputs.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="1-based start index within input file.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="1-based end index within input file (inclusive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run even if output folder already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned batch work without running.",
    )
    parser.add_argument(
        "--csv-charge",
        default="",
        help="Charge value to write into summary CSV.",
    )
    parser.add_argument(
        "--csv-mult",
        default="",
        help="Multiplicity value to write into summary CSV.",
    )
    parser.add_argument(
        "--library",
        "-l",
        type=str,
        help="Library for generating conformers of rough TS guess",
        default="rdkit",
    )
    parser.add_argument(
        "--calculator",
        "-ca",
        type=str,
        help="Using calculator for constraint optimization",
        default="orca",
    )
    parser.add_argument(
        "--num_conformer",
        "-nc",
        type=int,
        help="Number of precomplex conformations",
        default=3,
    )
    parser.add_argument(
        "--scan_num_relaxation",
        "-snr",
        type=int,
        help="Number of relaxation during each scan",
        default=3,
    )
    parser.add_argument(
        "--num_relaxation",
        "-nr",
        type=int,
        help="Number of relaxation during each scan",
        default=7,
    )

    parser.add_argument(
        "--ts_scale",
        "-ts",
        type=float,
        help="Scaling factor for bonds participating reactions",
        default=1.33,
    )
    parser.add_argument(
        "--form_scale",
        "-fs",
        type=float,
        help="Scaling factor for bond formation",
        default=1.1,
    )
    parser.add_argument(
        "--break_scale",
        "-bs",
        type=float,
        help="Scaling factor for bond dissociation",
        default=1.8,
    )
    parser.add_argument(
        "--num_process",
        "-np",
        type=int,
        help="Number of cores ",
        default=1,
    )
    parser.add_argument(
        "--protect_scale",
        "-ps",
        type=float,
        help="Scaling factor for protecting existing bonds",
        default=0.90,
    )
    parser.add_argument(
        "--max_separation",
        "-ms",
        type=float,
        help="Maximal distance separation for bond dissociation",
        default=3.0,
    )

    parser.add_argument(
        "--step_size",
        "-s",
        type=float,
        help="Step size for constraint optimization",
        default=0.15,
    )
    parser.add_argument(
        "--qc_step_size",
        "-qc",
        type=float,
        help="Maximal displacement change during QC calculation",
        default=0.10,
    )
    parser.add_argument(
        "--scan_qc_step_size",
        "-sqc",
        type=float,
        help="Number of relaxation during each scan",
        default=0.1,
    )

    parser.add_argument(
        "--energy_criteria",
        "-ec",
        type=float,
        help="Energy criteria for precomplex optimization",
        default=1.0,
    )
    parser.add_argument(
        "--preoptimize",
        "-p",
        type=int,
        help='Whehter to perform UFF optimization before gradual optimization',
        default=1,
    )
    parser.add_argument("--chg", type=int, help="Total charge of the system", default=0)
    parser.add_argument(
        "--mult", type=int, help="Total multiplicity of the system", default=1
    )
    parser.add_argument(
        "--use_crest",
        "-uc",
        type=int,
        help="Whether to sample several conformations with the CREST algorithm",
        default=0,
    )
    parser.add_argument(
        "--stereo_enumerate",
        "-se",
        type=int,
        help="Whether consider several stereoisomers",
        default=0,
    )

    parser.add_argument(
        "--maximal_displacement",
        "-md",
        type=float,
        help="Maximal displacement change for ",
        default=10000,
    )

    parser.add_argument(
        "--k",
        type=float,
        help="Force constant in UFF optimization",
        default=50,
    )

    parser.add_argument(
        "--window",
        "-w",
        type=float,
        help="Screening criteria for removing same pseudo-TS",
        default=0.5,
    )

    parser.add_argument(
        "--use_h_content",
        "-uh",
        type=int,
        help="Screening criteria for removing same pseudo-TS",
        default=1,
    )
    
    parser.add_argument(
        "--check_connectivity",
        "-cc",
        type=int,
        help="Screening criteria for removing same pseudo-TS",
        default=1,
    )

    parser.add_argument(
        "--check_stereo",
        "-cs",
        type=int,
        help="Screening criteria for removing same pseudo-TS",
        default=0,
    )


    if argv is None:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if args.input_file:
        return run_batch(args)
    if args.reaction is None:
        parser.print_help()
        return
    reaction_arg = args.reaction
    save_directory = (
        os.path.abspath(args.save_directory) if args.save_directory is not None else None
    )
    working_directory = (
        os.path.abspath(args.working_directory)
        if args.working_directory is not None
        else None
    )
    run_reaction(reaction_arg, args, save_directory, working_directory)


if __name__ == "__main__":
    main()
