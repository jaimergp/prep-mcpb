#!/usr/bin/env python
# encoding: utf-8

"""
Prepare files for MCPB.py using (py)Chimera
===========================================

We need to pay special attention to these details:

- Use PDB format 3.0
- Use Amber nomenclature for protein residues (HID, HIE, HIP, HIS and so on) and atom names (`HA12` instead of `2HA1`).
- Do not repeat atom names within the same residue (common in ligands)
- Use CAPITALIZED names (as in ZN, but not Zn or zN) for residue AND atom names
- The structures should contain all needed hydrogens
- ISOLATED metal ions and non-standard residues should be embedded in separate residues each. This also means that they should have unique residue and atom names.
- The total charge must be accurately specified before charge fitting!

By the end of this stage one should have a PDB file with the correct residues (separated, good names) and one antechamber-generated mol2+frcmod files for each non-standard residue (containing accurate atom types and charges).
"""

from __future__ import print_function, unicode_literals

try:
    from pychimera import patch_environ, load_chimera
    patch_environ()
    load_chimera()
except ImportError:
    raise ImportError("This script needs PyChimera. Install it with `conda install -c insilichem pychimera`.")

try:
    import pdb4amber
except ImportError:
    raise ImportError("This script needs AmberTools. Install it with `conda install -c AmberMD ambertools`.")


from argparse import ArgumentParser
from contextlib import contextmanager
from subprocess import check_call, CalledProcessError
from textwrap import dedent
from fileinput import input as fileinput
import os
import random
import string
import sys


import chimera
from SplitMolecule.split import split_molecule
from AddCharge import estimateNetCharge


__version__ = "0.0.1"
__author__ = "Jaime Rodríguez-Guerra"


def prepare_molecule_files(structure_query, interactive=True, charge_method='bcc'):
    structure = load_structure(structure_query)
    metal, protein, residues = split(structure)
    protein.basename = 'protein'
    prepared_protein = prepare_protein_amber(protein)

    if interactive:  # Ask for charges
        print('Guessing charges...')
        metal.charge = ask_for_charge(metal)
        for residue in residues:
            residue.charge = ask_for_charge(residue)

    metal.basename = metal.atoms[0].element.name.upper()
    prepared_metal = parameterize(metal, net_charge=None)
    prepared_residues = []

    for residue in residues:
        residue.basename = residue.residues[0].type
        parameterized = parameterize(residue, net_charge=residue.charge, charge_method=charge_method)
        prepared_residues.append(parameterized)

    return {
        'protein': {'pdb': prepared_protein},
        'metal': prepared_metal,
        'residues': prepared_residues
    }


def load_structure(query, reduce=True, remove_solvent=True):
    """
    Load a structure in Chimera. It can be anything accepted by `open` command.
    """
    print('Opening', query)
    chimera.runCommand('open ' + query)
    m = chimera.openModels.list()[0]
    m.setAllPDBHeaders({})

    if remove_solvent:
        print('  Removing solvent...')
        chimera.runCommand('del solvent')
    if reduce:
        print('  Adding hydrogens...')
        chimera.runCommand('addh')
    return m


def split(structure, interactive=True):
    """
    Split a structure into independent residues as expected by MCPB.py

    Parameters
    ==========
    structure : chimera.Molecule

    Returns
    =======
    protein, metal : chimera.Molecule
    residues : list of chimera.Molecule
        Non-standard residues
    """
    print('Preparing substructures...')
    metal = detect_metal_ions(structure, interactive=interactive)        
    pieces = split_molecule(structure, chains=None, ligands=True, connected=None, atoms=[[metal]])
    chimera.openModels.close([structure])
    chimera.openModels.add(pieces)
    pieces[0].name = pieces[0].name[:-1] + pieces[0].atoms[0].element.name
    pieces[0].basename = 'metal'
    pieces[0].residues[0].type = pieces[0].atoms[0].element.name.upper()
    pieces[1].name = pieces[1].name[:-1] + 'Protein'
    pieces[1].basename = 'protein'
    # Detect repeated names for different residues
    res_by_name = {}
    for piece in pieces[2:]:
        resname = piece.residues[0].type
        if resname not in res_by_name:
            res_by_name[resname] = piece
        else:  # repeated name!
            while True:
                new_name = _random_residue_type()
                if new_name not in res_by_name:
                    piece.name = piece.name.replace(piece.residues[0].type, new_name)
                    piece.residues[0].type == new_name
                    res_by_name[new_name] = piece
                    break

    print('  Identified following substructures:')
    for piece in pieces:
        print('   ', piece.name)
    # metal, protein, residues
    return pieces[0], pieces[1], pieces[2:]


def ask_for_charge(molecule):
    """
    Ask the user for the correct charge but providing an estimated value
    """
    estimated = estimateNetCharge(molecule.atoms)
    while True:
        answer = raw_input('  Specify charge for {} [{}]: '.format(molecule.name, estimated))
        if not answer:
            answer = estimated
        try:
            answer = int(answer)
            break
        except ValueError:
            print('  Provide a valid charge value (must be integer)')
    return answer


def detect_metal_ions(molecule, interactive=True, remove_others=True):
    """
    Find relevant metal ions in a molecule. By relevant we mean that
    they are object of study. So, if more than one metal ion is present, we
    should identify the one with a higher coordination number to get rid of
    mere solvent artefacts.

    Parameters
    ==========
    molecule : chimera.Molecule

    Returns
    =======
    metal : chimera.Atom
        Metal ion with higher relevance
    score : tuple
        Coordination number
    """
    metals = [a for a in molecule.atoms if a.element.isMetal]
    if interactive:
        metals.sort(key=lambda a: len(a.pseudoBonds), reverse=True)
        msg = '\n'.join(['  {:>3d}) {} ({} coordination bonds)'.format(i, a, len(a.pseudoBonds)) for (i, a) in enumerate(metals)])
        while True:
            choice = raw_input('  There are several metal ions present!\n' + msg + '\n' + '  Choose one [0]:  ')
            if not choice:
                choice = '0'
            if choice.isdigit() and 0 <= int(choice) < len(metals):
                break
            else:
                print('  Please provide a valid number!')
        metal = metals[int(choice)]
        print('    Using', metal, '... Any other metals will be removed!')
        for a in metals:
            if a is not metal:
                molecule.deleteAtom(a)
        return metal
    elif len(metals) == 1:
        return metals[0]
    else:
        raise ValueError('More than one metal present: {}'.format(
                         ', '.join([str(a) for a in metals])))


def parameterize(molecule, reduce=False, net_charge='auto', charge_method='bcc', atom_type='gaff'):
    """
    Add charges to a molecule using Antechamber and Parmchk

    Parameters
    ----------
    molecule : chimera.Molecule
    reduce : bool
        Add hydrogens to molecule before parameterization (in place)
    net_charge : int or 'auto' or None
        Total charge for the molecule. If 'auto' will try to guess, but can be wrong.
        If None, no charges will be computed (needed for metal ions)
    charge_method : str, default='bcc'
        Method used by -c option in antechamber. Available options:
        resp, bcc, cm2, esp, mul, gas
    

    Returns
    -------
    A dict with keys:
        molecule : chimera.Molecule
            Copy of the original molecule with new attributes
        mol2 : str
            Contents of antechamber-generated mol2 file
        frcmod : str
            Contents of parmchk-generated frcmod file
    """
    print('Preparing', molecule.name, '...')
    if net_charge is None:
        is_metal = True
        options = {}
    else:
        is_metal = False
        if net_charge == 'auto':
            net_charge = estimateNetCharge(molecule.atoms)
        options = {'net_charge': net_charge, 'charge_method': charge_method}
    
    for b in molecule.bonds:
        molecule.deleteBond(b)
    inpdb = molecule.basename + '.pdb'
    chimera.pdbWrite([molecule], molecule.openState.xform, inpdb)

    # Path element name column for metals
    if is_metal:
        with open(inpdb, 'r+') as f:
            lines = []
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    line = line.upper()
                lines.append(line)
            f.seek(0)
            f.write(''.join(lines))

    antechamber = 'antechamber -fi pdb -fo mol2 -i INPUT -o OUTPUT -at {atom_type}'.format(
                    atom_type=atom_type).split()
    if options:
        antechamber.extend('-c {charge_method} -nc {net_charge}'.format(**options).split())
    antechamber[6] = inpdb
    antechamber[8] = molecule.basename + '.mol2'
    print('  CMD:', ' '.join(antechamber))
    with open('antechamber_{}.log'.format(molecule.basename), 'w') as f:
        try:
            check_call(antechamber, stdout=f, stderr=f)
        except OSError:
            raise KnownError("  !!! ERROR - antechamber could not be located. Have you installed ambertools?")
        except CalledProcessError:
            raise KnownError('  !!! ERROR - Check antechamber_{}.log'.format(molecule.basename))

    if options:
        parmchk = 'parmchk2 -i INPUT -o OUTPUT -f mol2'.split()
        parmchk[2] = antechamber[8]
        parmchk[4] = os.path.splitext(inpdb)[0] + '.frcmod'
        print('  CMD:', ' '.join(parmchk))
        with open('parmchk2_{}.log'.format(molecule.basename), 'w') as f:
            try:
                check_call(parmchk, stdout=f, stderr=f)
            except OSError:
                raise KnownError("  !!! ERROR - parmchk2 could not be located. Have you installed ambertools?")
            except CalledProcessError:
                raise KnownError('  !!! ERROR - Check parmchk2_{}.log'.format(molecule.basename))

    result = {'pdb': antechamber[6], 'mol2': antechamber[8]}
    if is_metal:  
        result['restype'] = molecule.residues[0].type
        # add charge to mol2
        with open(antechamber[8], 'r+') as f:
            lines = []
            for line in f:
                if line.startswith('@<TRIPOS>ATOM'):
                    lines.append(line)  # add current one before skipping
                    line = next(f)
                    fields = line.split()
                    if fields[5] != fields[1]:
                        line = line.replace(fields[5], fields[1])
                    line = line.replace(fields[-1], str(float(molecule.charge)))
                lines.append(line)
            f.seek(0)
            f.write(''.join(lines))
    else:
        result['frcmod'] = parmchk[4]
    return result


def prepare_protein_amber(protein, ph=7):
    """
    Prepare molecule to contain correct residues, capping groups, and so on,
    using pdb4amber

    Parameters
    ==========
    molecule : chimera.Molecule

    Returns
    =======
    pdb
    """
    print('Preparing', protein.name, '...')
    inpdb = protein.basename + '.unfixed.pdb'
    chimera.pdbWrite([protein], protein.openState.xform, inpdb)
    pdb4amber.run(arg_pdbin=inpdb, arg_pdbout=protein.basename + '.pdb',
                  arg_reduce=True)
    return protein.basename + '.pdb'


def prepare_mcpb_input(structures, software_version='g09', cut_off=2.8):
    """
    Get all the substructure files and prepare the MCPB.py input
    """
    template = dedent("""
    original_pdb master.pdb
    group_name {name}
    cut_off {cut_off}
    ion_ids {metal_id}
    ion_mol2files {metal_mol2}
    naa_mol2files {residues_mol2}
    frcmod_files {residues_frcmod}
    large_opt {large_opt}
    software_version {software_version}
    """)

    # First collect all files in the same master PDB

    with open('master.unfixed.pdb', 'w') as f:
        pdbfiles = [structures['protein']['pdb'], structures['metal']['pdb']] + \
                   [r['pdb'] for r in structures['residues']]
        for line in fileinput(pdbfiles):
            f.write(line)

    # Fix residue numbering issues
    pdb4amber.run(arg_pdbin='master.unfixed.pdb', arg_pdbout='master.pdb')
    
    # Find metal ID
    for line in fileinput('master.pdb'):
        if line[17:21].strip() == structures['metal']['restype']:
            metal_id = int(line[6:12])

    name = os.path.basename(os.getcwd())
    with open('mcbp.in', 'w') as f:
        f.write(template.format(
            name=name, 
            metal_id=metal_id, 
            metal_mol2=structures['metal']['mol2'],
            residues_mol2=' '.join([r['mol2'] for r in structures['residues']]),
            residues_frcmod=' '.join([r['frcmod'] for r in structures['residues']]),
            cut_off=cut_off,
            software_version=software_version,
            large_opt=1
        ))
    
    return 'mcbp.in'


def parse_cli():
    p = ArgumentParser()
    p.add_argument('structure', 
        help='Structure to load. Can be a file or a identifier compatible with Chimera '
             'open command (e.g. pdb:4zf6)')
    p.add_argument('-p', '--path', 
        help='Directory that will host all generated files. If it does not exist, it will '
             'be created. If not provided, a 5-letter random string will be used.')
    p.add_argument('--chargemethod', choices='resp bcc cm2 esp mul gas'.split(),
        default='bcc', help='Charge method to use with antechamber. Default is bcc.')
    p.add_argument('--cutoff', default=2.8, type=float,
        help='Cutoff (in A) used by MCPB.py to build small and large models. Default is 2.8. '
             'Feel free to edit the resulting `mcpb.in` to change other parameters.')
    args = p.parse_args()
    if args.path is None:
        args.path = ''.join([random.choice(string.ascii_letters) for i in range(5)])
    return args


def main():
    args = parse_cli()
    with change_working_dir(args.path):
        structures = prepare_molecule_files(args.structure, charge_method=args.chargemethod)
        prepare_mcpb_input(structures, cut_off=args.cutoff)
    return args


class KnownError(BaseException):
    pass


@contextmanager
def change_working_dir(path):
    old_dir = os.getcwd()
    try:
        os.makedirs(path)
    except (IOError, OSError):
        pass
    os.chdir(path)
    yield
    os.chdir(old_dir)


def _random_residue_type():
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(3))


if __name__ == "__main__":
    try:
        print('prep-mcpb.py v{}. By @jaimergp, 2019.'.format(__version__))
        args = main()
    except KnownError as e:
        print(e)
        sys.exit(1)
    else:
        print('*' * 50)
        print('The preparation has ended successfully!')
        print('A mcpb.in file has been generated for your convenience.')
        print('Feel free to edit some parameters if needed (e.g. large_opt 1, etc).')
        print('You should be able to cd into `{}` and run the first MCPB.py step:'.format(args.path))
        print('    cd', args.path)
        print('    MCPB.py -i mcbp.in -s 1')
        print(', which should generate G09 input files to be processed with:')
        print('    MCPB.py -i mcbp.in -s 2')
        print('...and so on. Good luck!')
        print('*' * 50)


