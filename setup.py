#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import sys
import versioneer


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

version = versioneer.get_version()


setup(
    name='prep_mcpb',
    version=version,
    cmdclass=versioneer.get_cmdclass(),
    url='https://github.com/jaimergp/prep-mcpb',
    download_url='https://github.com/jaimergp/prep-mcpb/tarball/v' + version,
    license='MIT',
    author="Jaime Rodríguez-Guerra",
    author_email='jaime.rogue@gmail.com',
    description='Prepare a metal-containing structure for parameterization with MCPB.py using UCSF Chimera and AmberTools ',
    long_description=read('README.md'),
    packages=find_packages(),
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    install_requires='pychimera numpy scipy pdb4amber parmed'.split(),
    entry_points='''
        [console_scripts]
        prep_mcpb=prep_mcpb.prep_mcpb:main
        '''
)
