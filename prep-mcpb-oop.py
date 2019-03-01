

import os
import string
import shutil
from collections import defaultdict
from textwrap import dedent
from cStringIO import StringIO
from subprocess import check_call, CalledProcessError

import numpy as np

import chimera
from AddH import cmdAddH
from AddCharge import estimateNetCharge
from SplitMolecule.split import molecule_from_atoms, split_connected

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
                raise ValueError('{} is not empty!')
        self.root = root
        if prefix is None:
            prefix = os.path.basename(root)[:4]
        self.prefix = validate_short_id(prefix)

    def write(self, contents, filename, directory=None):
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
        filename = '{}_{}'.format(self.prefix, filename)
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
            self.name = validate_short_id(self._molecule.name[:6])
        # Create copy; original is stored at self._molecule
        self.molecule = cmdCombine([self._molecule], name=self.name, modelId=self._molecule.id, log=False)
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
            atoms = chimera.selection.OSLSelection(to_remove, models=[self.molecule])
        removed = []
        for a in atoms:
            if a in self.molecule.atoms:
                removed.append('{} serial {}'.format(a, a.serialNumber))
                self.molecule.removeAtom(a)
        return removed

    def add_h(self, **kwargs):
        kwargs.pop('molecules', None)
        cmdAddH(molecules=[self.molecule], **kwargs)

    def metals():
        return MetalResidueProvider(self.molecule)

    def ligands():
        return NonStandardResidueProvider(self.molecule)

    def protein():
        return ProteinProvider(self.molecule)

    def close():
        chimera.openModels.close([self._molecule, self.molecule])


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
            residues.append(MetalResidue(*cluster))
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
        clusters =[set([k] + v) for k, v in nearby.items()]
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


class MetalResidue(object):

    """
    Group of metal ions that are close to each other and can be
    tackled in the same QM calculation
    """

    def __init__(self, atoms, name=None):
        if name is None:
            name = atoms[0].element.name
        self.molecule = chimera.Molecule()
        self.name = self.molecule.name = validate_short_id(name)
        atoms_by_element = defaultdict(list)
        for a in atoms:
            atoms_by_element[a.element.name].append(a)

        for index, (element, atomlist) in enumerate(sorted(atoms_by_element.items()), 1):
            new_atom = self.molecule.newAtom(atom.name, atom.element, i)
            new_atom.charge = estimateNetCharge(new_atom)
            new_atom.setCoord(atom.coord())
            residue = self.molecule.newResidue(element.upper(), 'A', index, ' ')
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

    def to_pdb(self):
        """
        Export current molecule to a PDB file
        """
        s = StringIO()
        chimera.writePdb(self.molecule, self.molecule.openState.xform, s)
        s.seek(0)
        return s.read()


### LIGAND OBJECTS (non-protein, non-metal)

class NonStandardResidueProvider(object):

    def __init__(self, molecule):
        self.molecule = molecule

    def detect(self):
        """
        Detect connected residues

        TODO: Identify topologically identical pieces
        """
        candidate_atoms = chimera.evalSpec('~protein&~solvent',
                                           models=[self.molecule]).atoms()
        pieces_by_resname = defaultdict(list)
        for piece in split_connected(candidate_atoms):
            pieces_by_resname[piece.residues[0].type].append(piece)

        for resname, pieces in pieces_by_resname.items():
            if len(pieces) > 1:
                for i, piece in enumerate(pieces, 1):
                    pieces_by_resname[resname[:-1] + str(i)] = [piece]
                del pieces_by_resname[resname]

        residues = []
        for resname, piece in sorted(pieces_by_resname.items()):
            residues.append(NonStandardResidue(piece, name=resname))
        return residues


class NonStandardResidue(object):
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
    """

    def __init__(self, atoms, name=None, charge=None, extra_xyz=(,)):
        if name is None:
            name = atoms[0].residue.type
        self.name = validate_short_id(name)
        self.molecule = molecule_from_atoms(atoms[0].molecule, atoms, name=self.name)
        for i, xyz in enumerate(extra_xyz, 1):
            if len(xyz) != 3:
                raise ValueError('`extra_xyz` must be of shape (n, 3)')
            cs = self.molecule.newCoordSet(i)
            cs.load(xyz)
        if charge is None:
            charge = estimateNetCharge(charge)
        self.charge = charge

    def parameterize(self, atom_type='gaff', charge_method='bcc'):
        """
        Run antechamber and parmchk to obtain mol2 and frcmod files with
        residue parameters
        """
        pdb = self.name + '.pdb'
        mol2 = self.name + '.mol2'
        frcmod = self.name + '.frcmod'
        with enter_temporary_directory():
            with open(self.name, 'w') as f:
                f.write(self.to_pdb(replicas=False))

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
                    raise KnownError('  !!! ERROR - Check antechamber_{}.log'.format(molecule.basename))

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
                    raise KnownError('  !!! ERROR - Check parmchk2_{}.log'.format(molecule.basename))

            # Read files to return them back (tempdir will be erased at exit!)
            result = {}
            for key, path in (('pdb', pdb), ('mol2', mol2), ('frcmod', frcmod)):
                with open(path) as f:
                    result[key] = f.read()

        return result

    def to_pdb(self, replicas=True):
        """
        Export current molecule to a PDB file

        Parameters
        ==========
        replicas : bool, optional=True
            Whether to include all the copies of the ligand or
            only the primary one. Set to True when building the
            master, False for parameterization.
        """
        s = StringIO()
        # TODO: build a new molecule with all the residues, instead of using coordsets
        chimera.writePdb(self.molecule, self.molecule.openState.xform, s, replicas)
        s.seek(0)
        return s.read()



### PROTEIN OBJECTS

class ProteinProvider(object):
    pass


class Protein(object):
    pass


@contextmanager
def enter_temporary_directory(delete=True, **kwargs):
    old_dir = os.getcwd()
    tmpdir = tempfile.mkdtemp(**kwargs)
    os.chdir(path)
    yield
    os.chdir(old_dir)
    if delete:
        shutil.rmtree(path, ignore_errors=True)


def main():
    structure = RawStructure()
    structure.clean()
    structure.reduce()
    metals = structure.metals()
    ligands = structure.ligands()
    protein = structure.protein()