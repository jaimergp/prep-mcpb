
class MetalResidue(object):
    pass


class NonStandardResidue(object):
    pass


class Protein(object):
    pass


class StructurePreparer(object):

    def __init__():
        pass

    def clean():
        pass

    def reduce():
        pass

    add_hydrogens = reduce  #alias

    def metals():
        pass

    def ligands():
        pass

    def protein():
        pass


def main():
    structure = StructurePreparer()
    structure.clean()
    structure.reduce()
    metals = structure.metals()
    ligands = structure.ligands()
    protein = structure.protein()