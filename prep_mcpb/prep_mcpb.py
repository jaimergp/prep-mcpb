#!/usr/bin/env python
# encoding: utf-8

"""
Prepare files for MCPB.py using (py)Chimera
===========================================
"""

from __future__ import print_function
import os
import shutil
import string
from argparse import ArgumentParser
from collections import defaultdict, namedtuple, Counter
from contextlib import contextmanager
from cStringIO import StringIO
from itertools import combinations
from subprocess import check_call, CalledProcessError
from tempfile import mkdtemp
from textwrap import dedent

import numpy as np


try:
    from pychimera import patch_environ, load_chimera
    patch_environ()
    load_chimera()
    import chimera
    from AddCharge import estimateNetCharge
    from AddH import cmdAddH
    from chimera.specifier import evalSpec as chimera_selection
    from Combine import cmdCombine
    from SplitMolecule.split import molecule_from_atoms, split_connected
except ImportError:
    raise ImportError("prep_mcpb needs PyChimera. "
                      "Install it with `conda install -c insilichem pychimera`.")

try:
    import parmed as pmd
    from pdb4amber import AmberPDBFixer
except ImportError:
    raise ImportError("prep_mcpb needs AmberTools. "
                      "Install it with `conda install -c AmberMD ambertools`.")


__version__ = "0.0.2"
__author__ = "Jaime Rodríguez-Guerra"


### BASE OBJECTS

_valid_resname_characters = string.ascii_letters + string.digits + '-+'
_valid_filename_characters = '_()[] ' + _valid_resname_characters
def validate_short_id(s, max_length=6, valid=_valid_filename_characters):
    s = str(s)
    s = ''.join(c for c in s if c in valid).replace(' ', '_')
    if not s or len(s) > max_length:
        raise ValueError("ID must be str with {} characters maximum".format(max_length))
    return s


class MCPBWorkspace(object):
    """
    Handles file creation and organization

    Parameters
    ==========
    root : str
        Base directory that will store everything
    prefix : str
        Short identifier that will prefix all filenames
    reduce : bool
        Whether to use Amber's reduce to add hydrogens to protein only
    parameterize : bool
        Run antechamber + parmchk2 automatically in non-std residues
    """

    def __init__(self, root, prefix=None, reduce=True, parameterize=True):
        try:
            os.makedirs(root)
        except OSError:
            if os.listdir(root):
                raise ValueError('Directory `{}` is not empty!'.format(root))
        self.root = root
        if prefix is None:
            prefix = os.path.basename(root)[:4]
        self.prefix = validate_short_id(prefix[:6])
        self.reduce = reduce
        self.parameterize = parameterize

    def write(self, contents, filename, directory=None, prefix=None):
        """
        contents : file, file-like, path, str
            The contents of the file to write
        filename : str
            Suggested filename to be ammended for
            the workspace
        directory : str, optional
            If provided, a subdirectory of self.root.
            Will be created if it does not exist.
        """
        if hasattr(contents, 'read'):
            contents = contents.read()
        elif os.path.isfile(contents):
            with open(contents) as f:
                contents = f.read()
        # else contents is a str
        if prefix:
            filename = '{}{}'.format(self.prefix, filename)
        if directory:
            try:
                os.makedirs(os.path.join(self.root, directory))
            except OSError:
                pass
            path = os.path.join(self.root, directory, filename)
        else:
            path = os.path.join(self.root, filename)

        with open(path, 'w') as f:
            f.write(contents)

        return path

    def save(self, metals, nonstd_residues=(), protein=None):
        """
        Parameters
        ==========
        metals : list of MetalResidue, optional
            As provided by MetalResidueProvider.detect(). It can be
            set afterwards (before .save()).
        nonstd_residues : list of NonStandardResidue
            As provided by NonStandardResidueProvider.detect(). It can be
            set afterwards (before .save()).
        protein : Protein
            As provided by ProteinProvider.detect(). It can be
            set afterwards (before .save()).
        """
        pdbfiles = []
        ion_mol2files = []
        for center in metals:
            pdbfiles.append(center.to_pdb())
            for key, contents in center.to_mol2().items():
                self.write(contents, key + '.mol2')
                ion_mol2files.append(key + '.mol2')
        n_metals = sum([center.molecule.numAtoms for center in metals])
        ion_ids = range(1, n_metals+1)
        naa_mol2files, frcmod_files = [], []
        for ligand in nonstd_residues:
            pdbfiles.extend(ligand.to_pdb(conformers=True))
            if self.parameterize:
                files = ligand.parameterize()
                self.write(files['mol2'], ligand.name + '.mol2')
                self.write(files['frcmod'], ligand.name + '.frcmod')
                naa_mol2files.append(ligand.name + '.mol2')
                frcmod_files.append(ligand.name + '.frcmod')

        if protein:
            pdbfiles.append(protein.to_pdb(fix=True, add_h=self.reduce))

        self.write(self.master_pdb(pdbfiles), 'master.pdb')
        inputfile = self.mcpb_input(ion_ids=ion_ids, ion_mol2files=sorted(set(ion_mol2files)),
                                    naa_mol2files=naa_mol2files, frcmod_files=frcmod_files)
        self.write(inputfile, 'mcpb.in')

    @staticmethod
    def master_pdb(pdbfiles):
        s = StringIO()
        s.write("\n".join(pdbfiles))
        s.seek(0)
        pdbfixer = AmberPDBFixer(pmd.formats.PDBFile().parse(s))
        s = StringIO()
        pdbfixer.write_pdb(s)
        s.seek(0)
        s = s.read()
        return s

    def mcpb_input(self, ion_ids, ion_mol2files, naa_mol2files=(), frcmod_files=()):
        """
        ion_ids : list of int
            Serial numbers of each metal ion as present in the master PDB file
        ion_mol2files : list of str
            Filenames of each metal residue TYPE as generated with MetalResidue.to_mol2()
        naa_mol2files : list of str, optional, default=()
            Filenames of each non-standard residue (ligand) mol2 as generated with antechamber
        frcmod_files : list of str, optional, default=()
            Filenames of each non-standard residue (ligand) frcmod as generated with parmchk2
        """
        return dedent("""
            original_pdb master.pdb
            group_name {group_name}
            cut_off 2.8
            ion_ids {ion_ids}
            ion_mol2files {ion_mol2files}
            naa_mol2files {naa_mol2files}
            frcmod_files {frcmod_files}
            large_opt 1
            software_version g09
            """).format(group_name=self.prefix,
                        ion_ids=' '.join(map(str, ion_ids)),
                        ion_mol2files=' '.join(ion_mol2files),
                        naa_mol2files=' '.join(naa_mol2files),
                        frcmod_files=' '.join(frcmod_files))


class RawStructure(object):
    """
    Structure handles the original system, previous to the
    splitting. It is cleaned and reduced before delegating
    the split to the ``*Providers``

    Parameters
    ==========
    query : str
        Anything to be handled by Midas.open(). This can be
        an actual path in disk, or a database ID (e.g pdb:4okl).
        Only the first submodel will be considered.

    name : str, optional=MOL
        A short identifier that will act as a prefix in the
        workspace files.

    Attributes
    ==========
    molecule : chimera.Molecule
    """

    def __init__(self, query, name=None):
        _previously_opened = len(chimera.openModels.list(modelTypes=[chimera.Molecule]))
        chimera.runCommand('open ' + query)
        self._molecule = chimera.openModels.list(modelTypes=[chimera.Molecule])[_previously_opened]
        if name is None:
            name = validate_short_id(self._molecule.name[:6])
        self.name = name
        # Create copy; original is stored at self._molecule
        cmdCombine([self._molecule], name=self.name, modelId=self._molecule.id, log=False)
        self.molecule = chimera.openModels.list(modelTypes=[chimera.Molecule])[_previously_opened+1]
        chimera.openModels.remove([self._molecule])  # hide it from model panel

    def strip(self, to_remove='solvent&~@/pseudoBonds'):
        """
        Remove undesired atoms

        Parameters
        ==========
        to_remove : str or list of chimera.Atom
            A OSL selection query (e.g. solvent or :WAT) or a list
            of atoms to be removed
        """
        if isinstance(to_remove, basestring):  # assume is a chimera DSL
            atoms = chimera_selection(to_remove, models=[self.molecule]).atoms()
        removed = []
        for a in atoms:
            if a in self.molecule.atoms:
                removed.append('{} serial {}'.format(a, a.serialNumber))
                self.molecule.deleteAtom(a)
        return removed

    def add_h(self, **kwargs):
        kwargs.pop('molecules', None)
        cmdAddH(molecules=[self.molecule], **kwargs)

    def metals(self, **kwargs):
        return MetalResidueProvider(self.molecule, **kwargs).detect()

    def nonstd_residues(self, **kwargs):
        return NonStandardResidueProvider(self.molecule).detect(**kwargs)

    def protein(self, **kwargs):
        return ProteinProvider(self.molecule, **kwargs).detect()

    def close(self, **kwargs):
        chimera.openModels.close([self._molecule, self.molecule])


class PDBExportable(object):

    def __init__(self, molecule):
        self.molecule = molecule

    def to_pdb(self, molecule=None, **kwargs):
        """
        Export current molecule to a PDB file
        """
        if molecule is None:
            molecule = self.molecule
        s = StringIO()
        chimera.openModels.add([molecule])
        chimera.pdbWrite([molecule], molecule.openState.xform, s, **kwargs)
        chimera.openModels.remove([molecule])
        s.seek(0)
        return s.read()


### METAL OBJECTS

class MetalResidueProvider(object):

    def __init__(self, molecule, cluster_cutoff=3.5):
        self.molecule = molecule
        self.cluster_cutoff = 3.5

    def detect(self):
        # 1. Scan for atoms whose element isMetal, group by element
        metals = [a for a in self.molecule.atoms if a.element.isMetal]
        clusters = self.detect_clusters(metals)
        residues = []
        for cluster in clusters:
            residues.append(MetalResidue(cluster))
        return residues

    def detect_clusters(self, metals):
        """
        Detect groups of ions that are close to each other,
         `self.cluster_cutoff` A maximum.

        TODO: This can be probably done more efficiently
        """
        nearby = {}
        metals2 = metals[:]
        # 1. Iterate over all metals looking for neighbors within cutoff
        while metals2:
            m = metals2.pop()
            nearby[m] = []
            for n in metals:
                if m is n or n in nearby:
                    continue
                if m.coord().distance(n.coord()) < self.cluster_cutoff:
                    nearby[m].append(n)

        # 2. Iterate over previous sets trying to merge them if they
        #    have at least two atoms within cutoff
        clusters = [set([k] + v) for k, v in nearby.items()]
        merged = []
        clusters2 = clusters[:]
        for seed in clusters2:
            for i, cluster in enumerate(clusters):
                if i in merged:
                    continue
                min_dist = min([p.coord().distance(q.coord())
                                for p in cluster for q in seed
                                if p is not q] or [self.cluster_cutoff])
                if min_dist < self.cluster_cutoff:
                    seed.update(cluster)
                    merged.append(i)
        sorted_clusters = sorted(clusters, key=len, reverse=True)

        # 3. Discard subsets!
        saved = sorted_clusters[:1]
        for d in sorted_clusters:
            for c in saved:
                if d in saved or d == c or d.issubset(c):
                    break
            else:
                saved.append(d)
        return sorted(saved, key=len, reverse=True)


class MetalResidue(PDBExportable):

    """
    Group of metal ions that are close to each other and can be
    tackled in the same QM calculation
    """

    def __init__(self, atoms, name=None):
        atoms = list(atoms)
        if name is None:
            name = atoms[0].element.name
        self.molecule = chimera.Molecule()
        self.name = self.molecule.name = validate_short_id(name)
        atoms_by_element = defaultdict(list)
        for a in atoms:
            atoms_by_element[a.element.name].append(a)
        atomindex = 0
        for resindex, (element, atomlist) in enumerate(sorted(atoms_by_element.items()), 1):
            oldres = atomlist[0].residue
            residue = self.molecule.newResidue(element.upper(), oldres.id.chainId, oldres.id.position, ' ')
            for atom in atomlist:
                atomindex += 1
                new_atom = self.molecule.newAtom(atom.element.name.upper(), atom.element, atomindex)
                new_atom.charge = estimateNetCharge([new_atom])
                new_atom.setCoord(atom.coord())
                residue.addAtom(new_atom)

    @property
    def charge(self):
        return sum(a.charge for a in self.molecule.atoms)

    @property
    def elements(self):
        return set([a.element.name for a in self.molecule.atoms])

    def to_mol2(self):
        """
        Return mol2 files corresponding to all the residues present.
        Since this file is only needed to read the charge, element, and type
        we do not actually care about putting all the atoms in there. One is enough.
        """
        template = dedent("""
        @<TRIPOS>MOLECULE
        {element}
            1     0     1     0     0
        SMALL
        Charge estimated by Chimera's AddCharge.estimateNetCharge


        @<TRIPOS>ATOM
            1 {element}          {xyz[0]}    {xyz[1]}    {xyz[2]} {element}       {resid} {element}        {charge}
        @<TRIPOS>BOND
        @<TRIPOS>SUBSTRUCTURE
            1 {element}          1 TEMP              0 ****  ****    0 ROOT
        ROOT
        """)
        files = {}
        for r in self.molecule.residues:
            a = r.atoms[0]
            files[r.type] = template.format(
                element=r.type,
                xyz=a.coord().data(),
                charge=a.charge,
                resid=r.id.position)
        return files


### LIGAND OBJECTS (non-protein, non-metal)

class NonStandardResidueProvider(object):

    def __init__(self, molecule, **kwargs):
        self.molecule = molecule
        self._kwargs = kwargs

    def detect(self, **kwargs):
        """
        Detect connected residues

        TODO: Identify topologically identical pieces
        """
        # Find anything that is not a protein nor solvent,
        # but do choose solvent with pseudobonds (metal-coordinated)
        atoms = chimera_selection('~protein&~solvent|solvent&@/pseudoBonds',
                                  models=[self.molecule]).atoms()
        candidate_atoms = [a for a in atoms if not a.element.isMetal]
        pieces_by_resname = defaultdict(list)
        for i, atoms in split_connected(candidate_atoms):
            if len(atoms) == 1:
                print('! Warning: nonstd residue of 1 atom will be ignored', str(atoms[0]))
            else:
                pieces_by_resname[atoms[0].residue.type].append(atoms)

        pieces_by_topology = self.detect_topologies([p for pieces in pieces_by_resname.values()
                                                     for p in pieces])

        residues = []
        print('Detected', len(pieces_by_topology), 'non-std residues:')
        for resname, piece in sorted(pieces_by_topology.items()):
            print(' ', resname, 'of', len(piece[0]), 'atoms, has', len(piece[1:]), 'replicas')
            for replica in piece[1:]:
                print('  ', len(replica))
            residues.append(NonStandardResidue(piece[0], name=resname, extra_xyz=piece[1:], **kwargs))
        return residues

    def detect_topologies(self, pieces):
        """
        Detect topologically equivalent pieces in a given list
        of groups of connected atoms (residues)

        Parameters
        ==========
        pieces : list of list of chimera.Atom
            Each sublist of chimera.Atom is considered a connected residue

        Returns
        =======
        pieces_by_topology : dict
            Mapping of residue name to list of groups of atoms
        """
        pieces_by_topology = defaultdict(list)
        pieces = sorted(pieces, key=len, reverse=True)
        # Seed it with the biggest structure
        pieces_by_topology[pieces[0][0].residue.type].append(pieces[0])
        for p in pieces[1:]:
            for tname, tops in pieces_by_topology.items():
                if self._same_topology(p, tops[0]):
                    pieces_by_topology[tname].append(p)
                    break
            else:
                i = 0
                pname = p[0].residue.type
                while pname in pieces_by_topology:
                    pname = pname[:-1] + str(i)
                    i += 1
                pieces_by_topology[pname].append(p)
        return pieces_by_topology

    @staticmethod
    def _same_topology(a, b):
        """
        Test if two groups of atoms are topologically equivalent

        Parameters
        ==========
        a, b : list of chimera.Atoms
        """
        # Same number of atoms?
        if len(a) != len(b):
            return False
        # Same formula?
        if Counter([at.element.name for at in a]) != Counter([at.element.name for at in b]):
            return False
        # Same residue names? (give the user an opportunity to separate them manually)
        if Counter([at.residue.type for at in a]) != Counter([at.residue.type for at in b]):
            return False
        # TODO: test bond graph (connectivity)
        # for now we are trusting that isomers are labeled differently
        return True


class NonStandardResidue(PDBExportable):
    """
    A non-metal, non-protein connected molecule that must
    be parameterized with antechamber.

    Parameters
    ==========
    atoms : list of chimera.Atom
        Atoms that will build the final residue. They should be part
        of the same chimera.Residue.
    name : str, optional
        A short identifier
    charge : int, optional
        Charge of the residue. Will be guesstimated if not provided.
    extra_xyz : list of 3-tuple of float
        Extra coordinates for the same compound. This is, if a residue
        is present several times in the same structure, parameterize
        only once but build the final PDB with all the extra
        positions.
    atom_type : str, optional=gaff
        Used by antechamber.
    charge_method : str, optional=bcc
        Used by antechamber. Choose between
    rename_atoms : bool, optional=True
        If ``extra_xyz`` molecules are provided, atom names should match the main one. Setting
        this to True will enforce that rule.
    """
    _charge_methods = 'resp bcc cm2 esp mul gas'.split()
    _atom_types = 'gaff gaff2'.split()
    def __init__(self, atoms, name=None, charge=None, extra_xyz=(), atom_type='gaff', charge_method='bcc',
                 rename_atoms=True):
        if name is None:
            name = atoms[0].residue.type
        self.name = validate_short_id(name, max_length=3, valid=_valid_resname_characters)
        self.rename_atoms = rename_atoms

        self.molecule = self.molecular_residue_from_atoms(atoms, rename_atoms=self.rename_atoms)
        if not all([len(m) == len(atoms) for m in extra_xyz]):
            raise ValueError("Extra molecules for {} must have same number of atoms "
                             "as the main one ({})!".format(self.name, len(atoms)))
        else:
            self.extra_xyz = [self.molecular_residue_from_atoms(atoms, rename_atoms=self.rename_atoms, index=i)
                              for i, atoms in enumerate(extra_xyz, 2)]

        if charge is None:
            charge = estimateNetCharge(atoms)
        self.charge = charge
        if atom_type not in self._atom_types:
            raise ValueError('atom_type must be one of {}'.format(', '.join(self._atom_types)))
        self.atom_type = atom_type
        if charge_method not in self._charge_methods:
            raise ValueError('charge_method must be one of {}'.format(', '.join(self._charge_methods)))
        self.charge_method = charge_method

    def parameterize(self, atom_type=None, charge_method=None):
        """
        Run antechamber and parmchk to obtain mol2 and frcmod files with
        residue parameters
        """
        if atom_type is None:
            atom_type = self.atom_type
        if charge_method is None:
            charge_method = self.charge_method
        pdb = self.name + '.pdb'
        mol2 = self.name + '.mol2'
        frcmod = self.name + '.frcmod'
        with enter_temporary_directory(delete=False) as tmpdir:
            with open(pdb, 'w') as f:
                f.write(self.to_pdb(conformers=False))

            # Run antechamber
            options = {'atom_type': atom_type, 'charge_method': charge_method, 'net_charge': self.charge}
            cmd = ('antechamber -fi pdb -fo mol2 -i INPUT -o OUTPUT '
                   '-at {atom_type} -c {charge_method} -nc {net_charge}'.format(**options)).split()
            cmd[6] = pdb
            cmd[8] = mol2
            with open('{}-antechamber.log'.format(self.name), 'w') as f:
                try:
                    check_call(cmd, stdout=f, stderr=f)
                except OSError:
                    raise KnownError("  !!! ERROR - antechamber could not be located. "
                                    "Have you installed ambertools?")
                except CalledProcessError:
                    raise KnownError('  !!! ERROR - Check {}/{}-antechamber.log'.format(tmpdir, self.name))

            # Run parmchk
            cmd = 'parmchk2 -i INPUT -o OUTPUT -f mol2'.split()
            cmd[2] = mol2
            cmd[4] = frcmod
            with open('{}_parmchk2.log'.format(self.name), 'w') as f:
                try:
                    check_call(cmd, stdout=f, stderr=f)
                except OSError:
                    raise KnownError("  !!! ERROR - parmchk2 could not be located. "
                                    "Have you installed ambertools?")
                except CalledProcessError:
                    raise KnownError('  !!! ERROR - Check {}/{}-parmchk2.log'.format(tmpdir, self.name))

            # Read files to return them back (tempdir will be erased at exit!)
            result = {}
            for key, path in (('pdb', pdb), ('mol2', mol2), ('frcmod', frcmod)):
                with open(path) as f:
                    result[key] = f.read()

        return result

    def to_pdb(self, conformers=True, **kwargs):
        """
        Export current molecule to a PDB file

        Parameters
        ==========
        conformers : bool, optional=True
            Whether to include all the copies of the ligand or
            only the primary one. Set to True when building the
            master, False for parameterization.

        Returns
        =======
        str or list of str
            If conformers=True, list of str for PDB files for the main conformer
            and each variant. If False, str of PDB file of the main conformer.
        """
        if conformers:
            pdbfiles = [PDBExportable.to_pdb(self, **kwargs)]
            for extra in self.extra_xyz:
                pdbfiles.append(PDBExportable.to_pdb(self, molecule=extra, **kwargs))
            return pdbfiles
        return PDBExportable.to_pdb(self)

    def molecular_residue_from_atoms(self, atoms, rename_atoms=True, index=1):
        """
        Create a new chimera.Molecule instance from a set of atoms,
        guaranteeing that they will only constitute one chimera.Residue
        """
        m = chimera.Molecule()
        m.name = self.name
        r = m.newResidue(self.name, 'A', index, ' ')
        old2new = {}
        for serial, atom in enumerate(atoms, 1):
            new_atom = m.newAtom(atom.name, atom.element, serial)
            new_atom.setCoord(atom.coord())
            r.addAtom(new_atom)
            old2new[atom] = new_atom

        for bond in set([bond for a in atoms for bond in a.bonds]):
            if all(a in old2new for a in bond.atoms):
                b = m.newBond(old2new[bond.atoms[0]], old2new[bond.atoms[1]])

        if rename_atoms:
            element_names = defaultdict(int)
            for root in m.roots(True):
                for atom in m.traverseAtoms(root):
                    element_names[atom.element.name] += 1
                    atom.name = '{}{}'.format(atom.element.name.upper(), element_names[atom.element.name])
        return m


### PROTEIN OBJECTS

class ProteinProvider(object):

    def __init__(self, molecule, name=None):
        self.molecule = molecule
        if name is None:
            name = molecule.name.split()[0][:4]
        self.name = validate_short_id(name, max_length=4, valid=_valid_resname_characters)

    def detect(self):
        atoms = chimera_selection('protein', models=[self.molecule]).atoms()
        if atoms:
            return Protein(atoms, self.name)


class Protein(PDBExportable):

    def __init__(self, atoms, name=None):
        if name is None:
            name = atoms[0].molecule.name.split([0])[:4]
        self.name = validate_short_id(name, max_length=4, valid=_valid_resname_characters)
        self.molecule = molecule_from_atoms(atoms[0].molecule, atoms, name=self.name)

    def fix(self, add_h=True):
        s = StringIO()
        s.write(self.to_pdb())
        s.seek(0)
        pdbfixer = AmberPDBFixer(pmd.formats.PDBFile().parse(s))
        if add_h:
            pdbfixer.add_hydrogen()

        # Assign HIS/HID/HIE correctly
        pdbfixer.assign_histidine()
        # Fix CYS -> CYX if needed
        sslist, cys_cys_atomidx_set = pdbfixer.find_disulfide()
        pdbfixer.rename_cys_to_cyx(sslist)

        # Remove altLocs
        for atom in pdbfixer.parm.atoms:
            atom.altloc = ''
            for oatom in atom.other_locations.values():
                oatom.altloc = ''

        output = pdbfixer._write_pdb_to_stringio(
            cys_cys_atomidx_set=cys_cys_atomidx_set,
            disulfide_conect=True)

        output.seek(0)
        return output.read()

    def to_pdb(self, fix=False, **kwargs):
        if fix:
            return self.fix(**kwargs)
        return PDBExportable.to_pdb(self)


## UTILS

@contextmanager
def enter_temporary_directory(delete=True, **kwargs):
    old_dir = os.getcwd()
    path = mkdtemp(**kwargs)
    os.chdir(path)
    yield path
    os.chdir(old_dir)
    if delete:
        shutil.rmtree(path, ignore_errors=True)


class KnownError(BaseException):
    pass


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
    p.add_argument('--no_reduce', action='store_true',
        help='Do not try to add hydrogens to the molecule automatically.')
    p.add_argument('--no_parameterize', action='store_true',
        help='Do not run antechamber+parmchk in non-std residues '
             '(useful for debugging piece identification).')
    p.add_argument('--charge_method', choices='resp bcc cm2 esp mul gas'.split(),
        default='bcc', help='Charge method to use with antechamber. Default is bcc.')
    p.add_argument('--atom_type', choices='gaff gaff2'.split(),
        default='gaff', help='Atom types used by antechamber. Default is gaff.')
    p.add_argument('--cluster_cutoff', default=3.5, type=float,
        help='Cutoff (in A) used to find clusters of metals in the provided structure. '
             'It is the maximum distance a metal ion can be from other nearby. Default is 3.5.')
    p.add_argument('--mcpb_cutoff', default=2.8, type=float,
        help='Cutoff (in A) used by MCPB.py to build small and large models. Default is 2.8. '
             'Feel free to edit the resulting `mcpb.in` to change other parameters.')
    args = p.parse_args()
    if args.path is None:
        args.path = ''.join([random.choice(string.ascii_letters) for i in range(5)])
    return args


def main():
    args = parse_cli()
    name = validate_short_id(os.path.basename(args.path))
    # Load requested structure and do global fixes
    structure = RawStructure(args.structure, name=name)
    structure.strip(to_remove=args.strip)
    if not args.no_reduce:
        structure.add_h()

    # Discover components
    metals = structure.metals(cluster_cutoff=args.cluster_cutoff)
    nonstd_residues = structure.nonstd_residues(charge_method=args.charge_method, atom_type=args.atom_type)
    protein = structure.protein()

    # Write everything to the results directory
    workspace = MCPBWorkspace(args.path, prefix=name, reduce=not args.no_reduce,
                              parameterize=not args.no_parameterize)
    workspace.save(metals=metals, nonstd_residues=nonstd_residues, protein=protein)


if __name__ == "__main__":
    main()
