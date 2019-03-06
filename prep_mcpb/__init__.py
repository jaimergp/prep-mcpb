#!/usr/bin/env python
# encoding: utf-8

from .prep_mcpb import RawStructure, MCPBWorkspace, MetalResidue, MetalResidueProvider, \
                       NonStandardResidue, NonStandardResidueProvider, Protein, ProteinProvider

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
__author__ = "Jaime Rodríguez-Guerra"

