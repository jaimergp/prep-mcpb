#!/usr/bin/env python
# encoding: utf-8

"""
Prepare files for MCPB.py using (py)Chimera
===========================================

Current recipe:

1. Download or open the metal-containing structure. Remove waters, add hydrogens.
2. Look for metal centers and make the user choose one. Non-chosen are deleted.
3. Split it into protein, chosen metal and non-standard residues.
4. Prepare/fix protein with pdb4amber+reduce.
5. Estimate metal & ligand net charges with Chimera but allow the user to override them.
6. antechamber metal PDB to obtain MOL2 and patch it with correct atom type and charge.
7. antemchamber+parmchk the ligands.
8. Generate the full PDB (master.pdb) with correct numbers and case via pdb4amber.
"""

from __future__ import print_function, unicode_literals

try:
    from pychimera import patch_environ, load_chimera
    patch_environ()
    load_chimera()
except ImportError:
    raise ImportError("This script needs PyChimera. "
                      "Install it with `conda install -c insilichem pychimera`.")

try:
    import pdb4amber
except ImportError:
    raise ImportError("This script needs AmberTools. "
                      "Install it with `conda install -c AmberMD ambertools`.")


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


def prepare_molecule_files(structure_query, interactive=True, charge_method='bcc', strip='solvent'):
    structure = load_structure(structure_query, strip=strip)
    metals, residues, proteins = split(structure)

    if proteins:
        proteins[0].basename = 'protein'
        prepared_protein = {'pdb': prepare_protein_amber(proteins[0])}
    else:
        prepared_protein = {}

    print('Guessing charges...')
    for res in metals + residues:
        res.charge = ask_for_charge(res, interactive)

    prepared_metals = []
    for metal in metals:
        metal.basename = metal.name.split()[-1]
        metal.residues[0].type = metal.atoms[0].name = metal.atoms[0].element.name.upper()
        prepared_metals.append(parameterize(metal, net_charge=None))

    prepared_residues = []
    for residue in residues:
        residue.basename = residue.residues[0].type
        parameterized = parameterize(residue, net_charge=residue.charge, charge_method=charge_method)
        prepared_residues.append(parameterized)

    return {'metals': prepared_metals,
            'residues': prepared_residues,
            'protein': prepared_protein}


def load_structure(query, reduce=True, strip='solvent&~@/pseudoBonds'):
    """
    Load a structure in Chimera. It can be anything accepted by `open` command.

    Parameters
    ==========
    query : str
        Path to molecular file, or special query for Chimera's open (e.g. pdb:3pk2).
    reduce : bool
        Add hydrogens to structure. Defaults to True.
    strip : str
        Chimera selection spec that will be removed. Defaults to solvent&~@/pseudoBonds
        (solvent that is not attached to a metal ion).
    """
    print('Opening', query)
    chimera.runCommand('open ' + query)
    m = chimera.openModels.list()[0]
    m.setAllPDBHeaders({})

    if strip:
        print('  Removing {}...'.format(strip))
        chimera.runCommand('del ' + strip)
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
    # Separate metal ions from the rest
    metal_atoms = detect_metal_ions(structure, interactive=interactive)
    first_pieces = split_molecule(structure, chains=None, ligands=None, connected=None,
                                  atoms=[[a] for a in metal_atoms])
    metals, nonmetal = first_pieces[:-1], first_pieces[-1]  # metal ions are returned first
    chimera.openModels.close([structure])

    for metal in metals:
        metal.name = metal.name[:-1] + metal.atoms[0].name

    # Separate ligands from protein
    nonmetal.name = nonmetal.name[:-2]  # remove id added at split
    chimera.openModels.add([nonmetal])
    ligands, proteins = [], []
    second_pieces = split_molecule(nonmetal, chains=None, ligands=True,
                                   connected=None, atoms=None)

    if second_pieces:
        for p in second_pieces:
            (ligands if p.numResidues == 1 else proteins).append(p)
        chimera.openModels.close([nonmetal])
    else:  # only one type of molecule present (protein or nonstandard residue)
        (ligands if nonmetal.numResidues == 1 else proteins).append(nonmetal)

        # If only one residue, antechamber needs separate ones for each connected unit
        if ligands:
            third_pieces = split_molecule(nonmetal, chains=None, ligands=None,
                                          connected=True, atoms=None)
            if third_pieces:  # we did split the ligand in parts
                ligands = third_pieces
                chimera.openModels.close([nonmetal])
            else: # the ligand is a single unit, nonmetal is left untouched
        chimera.openModels.remove([nonmetal])
            # In this case, we need to patch the molecule name (CLI aesthetics)
            for ligand in ligands:
                ligand.name += ' ' + ligand.residues[0].type


    pieces = metals + ligands + proteins
    chimera.openModels.add(pieces)

    # Detect repeated names for different residues
    res_by_name = {}
    for mol in ligands:
        resname = mol.residues[0].type
        if resname not in res_by_name:
            res_by_name[resname] = mol
        else:  # repeated name!
            while True:
                new_name = _random_residue_type()
                if new_name not in res_by_name:
                    mol.name = mol.name.replace(mol.residues[0].type, new_name)
                    mol.residues[0].type = new_name
                    res_by_name[new_name] = mol
                    break

    print('  Identified following substructures:')
    for piece in pieces:
        print('    ', piece.name)
    return metals, ligands, proteins


def ask_for_charge(molecule, interactive=True):
    """
    Ask the user for the correct charge but providing an estimated value
    """
    estimated = estimateNetCharge(molecule.atoms)
    if not interactive:
        return estimated
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
        msg = '\n'.join(['  {:>3d}) {} ({} coordination bonds)'.format(i, a, len(a.pseudoBonds))
                         for (i, a) in enumerate(metals)])
        while True:
            s = raw_input('  There are several metal ions present!\n'
                          '  Optionally, provide a new residue name after a colon (e.g. 0:ZN)\n'
                          '  Several ones can be specified for the same metal center with '
                          'spaces (e.g. 0:ZN1 1:ZN2)\n' + msg + '\n  Choose now [0]:  ')
            try:
                choices = _parse_metal_choice(s, len(metals))
                chosen_metals = []
                for index, name in choices:
                    metal = metals[index]
                    if name is not None:
                        metal.name = name
                    chosen_metals.append(metal)
                names = [m.name for m in chosen_metals]
                if len(names) != len(set(names)):
                    raise ValueError('  Metal names contain repetitions. Please provide '
                                     'different names manually!')
            except ValueError as e:
                print(e)
            else:
                break
        print('    Using {}... Any other metals will be removed!'.format(
              ', '.join(map(str, chosen_metals))))
        for a in metals:
            if a not in chosen_metals:
                molecule.deleteAtom(a)
        return chosen_metals
    elif len(metals) == 1:
        return metals[0]
    else:
        raise ValueError('More than one metal present: {}'.format(
                         ', '.join([str(a) for a in metals])))


def _parse_metal_choice(s, max_value):
    """
    Parse user options provided in ``detect_metal_options``.

    The syntax is <position>,[<new name>], using semicolons to choose several ones.
    <new name> can only be 3-letters max and should not collide with existing Amber
    types. This is not checked, so be careful! If you choose several ones, they
    are considered part of the same metal center! Do not use it for unrelated ions;
    instead run the script several times and use the step 1n.
    For example:

        - 0  # would select the first one (default), without renaming
        - 0:ZN1 # select first one with a new name (ZN1)
        - 0:ZN1 1:ZN2 # select first and second with new names

    Parameters
    ==========
    s : str

    Return
    ======
    list of (index, name)
        name can be None
    """
    if not s:
        return [(0, None)]
    result = []
    for selection in s.split():
        name = None
        fields = selection.split(':')
        if len(fields) == 1:
            name == None
        elif len(fields) == 2 and 0 < len(fields[1]) <= 3:
            name = fields[1]
        else:
            raise ValueError('    !!! Wrong syntax!')
        index = int(fields[0])
        if index < 0 or index >= max_value:
            raise ValueError('    !!! Index must be within 0 and ' + str(max_value))
        result.append((index, name))
    return result


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
            raise KnownError("  !!! ERROR - antechamber could not be located. "
                             "Have you installed ambertools?")
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
                raise KnownError("  !!! ERROR - parmchk2 could not be located. "
                                 "Have you installed ambertools?")
            except CalledProcessError:
                raise KnownError('  !!! ERROR - Check parmchk2_{}.log'.format(molecule.basename))

    result = {'pdb': antechamber[6], 'mol2': antechamber[8]}
    if is_metal:   # fix charge and atom type in antechamber-generated mol2
        with open(antechamber[8], 'r+') as f:
            lines = []
            for line in f:
                if line.startswith('@<TRIPOS>ATOM'):
                    lines.append(line)  # add current one before skipping
                    line = next(f)
                    fields = line.split()
                    if fields[5] != fields[1]:  # fix atom type if it does not match resname
                        line = line.replace(fields[5], fields[1])
                    line = line.replace(fields[-1], str(float(molecule.charge)))  # replace charge
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
    large_opt 1
    software_version {software_version}
    """)

    # First collect all files in the same master PDB
    pdbfiles = [s['pdb'] for s in structures['metals'] + structures['residues']]
    if 'pdb' in structures['protein']:
        pdbfiles.append(structures['protein']['pdb'])

    with open('master.unfixed.pdb', 'w') as f:
        for line in fileinput(pdbfiles):
            f.write(line)

    # Fix residue numbering issues
    pdb4amber.run(arg_pdbin='master.unfixed.pdb', arg_pdbout='master.pdb')

    name = os.path.basename(os.getcwd())
    with open('mcbp.in', 'w') as f:
        f.write(template.format(
            name=name,
            metal_id=' '.join(map(str, range(1, len(structures['metals']) + 1))),
            metal_mol2=' '.join([s['mol2'] for s in structures['metals']]),
            residues_mol2=' '.join([r['mol2'] for r in structures['residues']]),
            residues_frcmod=' '.join([r['frcmod'] for r in structures['residues']]),
            cut_off=cut_off,
            software_version=software_version,
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
    p.add_argument('--strip', default='solvent&~@/pseudoBonds',
        help='Atoms to be removed from original structure. By default, only the solvent. '
             'Any query supported by UCSF Chimera atom-spec can be used. For example, '
             'it can be used to delete unneeded NMR models with ~#0.1.')
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
    if os.path.isfile(args.structure):
        args.structure = os.path.abspath(args.structure)
    with change_working_dir(args.path):
        structures = prepare_molecule_files(args.structure, strip=args.strip,
                                            charge_method=args.chargemethod)
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
        print('    MCPB.py -i mcbp.in -s 1n  # n -> do not rename metal ions')
        print(', which should generate G09 input files to be processed with:')
        print('    MCPB.py -i mcbp.in -s 2')
        print('...and so on. Good luck!')
        print('*' * 50)


