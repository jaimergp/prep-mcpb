#!/usr/bin/env python
# encoding: utf-8

"""
Prepare files for MCPB.py using (py)Chimera
===========================================
"""

from __future__ import print_function
try:
    from pychimera import patch_environ, load_chimera
    patch_environ()
    load_chimera()
except ImportError:
    raise ImportError("This script needs PyChimera. "
                      "Install it with `conda install -c insilichem pychimera`.")

try:
    from pdb4amber import AmberPDBFixer
    import parmed as pmd
except ImportError:
    raise ImportError("This script needs AmberTools. "
                      "Install it with `conda install -c AmberMD ambertools`.")

import os
import string
import shutil
from collections import defaultdict
from textwrap import dedent
from cStringIO import StringIO
from subprocess import check_call, CalledProcessError
from contextlib import contextmanager
from argparse import ArgumentParser
from tempfile import mkdtemp

import numpy as np

import chimera
from chimera.specifier import evalSpec as chimera_selection
from AddH import cmdAddH
from AddCharge import estimateNetCharge
from SplitMolecule.split import molecule_from_atoms, split_connected
from Combine import cmdCombine

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
    """

    def __init__(self, root, prefix=None):
        try:
            os.makedirs(root)
        except OSError:
            if os.listdir(root):
                raise ValueError('Directory `{}` is not empty!'.format(root))
        self.root = root
        if prefix is None:
            prefix = os.path.basename(root)[:4]
        self.prefix = validate_short_id(prefix)

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
            files = ligand.parameterize()
            pdbfiles.extend(ligand.to_pdb(conformers=True))
            self.write(files['mol2'], ligand.name + '.mol2')
            self.write(files['frcmod'], ligand.name + '.frcmod')
            naa_mol2files.append(ligand.name + '.mol2')
            frcmod_files.append(ligand.name + '.frcmod')

        if protein:
            pdbfiles.append(protein.to_pdb(fix=True))

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
                new_atom = self.molecule.newAtom(atom.name, atom.element, atomindex)
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

        for resname, pieces in pieces_by_resname.items():
            if len(pieces) > 1:
                # check if they are topologically equivalent
                # in the mean time, the number of atoms should suffice
                num_atoms = max([len(p) for p in pieces])  # biggest piece is considered the main
                for i, piece in enumerate(pieces):
                    if len(piece) != num_atoms:
                        new_resname = resname[:-1] + str(i+1)  # TODO: Check valid resname
                        pieces_by_resname[new_resname] = piece
                        pieces.pop(i)

        residues = []
        for resname, piece in sorted(pieces_by_resname.items()):
            for p in piece:
                print(p[0].residue.type, len(p))
            residues.append(NonStandardResidue(piece[0], name=resname, extra_xyz=piece[1:], **kwargs))
        return residues


class NonStandardResidue(PDBExportable):
    """
    A non-metal, non-protein connected molecule that must
    be parameterized with antechamber.

    Parameters
    ==========
    atoms : list of chimera.Atom
        Atoms that will build the final residue
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
    """
    _charge_methods = 'resp bcc cm2 esp mul gas'.split()
    _atom_types = 'gaff gaff2'.split()
    def __init__(self, atoms, name=None, charge=None, extra_xyz=(), atom_type='gaff', charge_method='bcc'):
        if name is None:
            name = atoms[0].residue.type
        self.name = validate_short_id(name)
        self.molecule = molecule_from_atoms(atoms[0].molecule, atoms, name=self.name)
        # for i, xyz in enumerate(extra_xyz, 1):
        #     if len(xyz) != 3:
        #         raise ValueError('`extra_xyz` must be of shape (n, 3)')
        #     cs = self.molecule.newCoordSet(i)
        #     cs.load(xyz)
        if not all([len(m) == len(atoms) for m in extra_xyz]):
            raise ValueError("Extra molecules must have same number of atoms as the main one!")
        else:
            self.extra_xyz = [molecule_from_atoms(at[0].molecule, at, name=self.name)
                              for at in extra_xyz]
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

    def to_pdb(self, conformers=True):
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
            pdbfiles = [PDBExportable.to_pdb(self)]
            for extra in self.extra_xyz:
                print(extra.name, extra.numAtoms)
                pdbfiles.append(PDBExportable.to_pdb(self, molecule=extra))
            return pdbfiles
        return PDBExportable.to_pdb(self)



### PROTEIN OBJECTS

class ProteinProvider(object):

    def __init__(self, molecule, name=None):
        self.molecule = molecule
        if name is None:
            name = molecule.name.split()[0][:4]
        self.name = validate_short_id(name)

    def detect(self):
        atoms = chimera_selection('protein', models=[self.molecule]).atoms()
        if atoms:
            return Protein(atoms, self.name)


class Protein(PDBExportable):

    def __init__(self, atoms, name=None):
        if name is None:
            name = atoms[0].molecule.name.split([0])[:4]
        self.name = validate_short_id(name)
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

    def to_pdb(self, fix=False):
        if fix:
            return self.fix()
        return PDBExportable.to_pdb(self)


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
    structure.add_h()

    # Discover components
    metals = structure.metals(cluster_cutoff=args.cluster_cutoff)
    nonstd_residues = structure.nonstd_residues(charge_method=args.charge_method, atom_type=args.atom_type)
    protein = structure.protein()

    # Write everything to the results directory
    workspace = MCPBWorkspace(args.path, prefix=name)
    workspace.save(metals=metals, nonstd_residues=nonstd_residues, protein=protein)


if __name__ == "__main__":
    main()
