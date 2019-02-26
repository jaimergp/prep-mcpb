# prep-mcpb.py

A *step 0* for your MCPB.py protocols!

This script uses UCSF Chimera and AmberTools to preprocess a metal-containing structure for `MCPB.py` parameterization by following the recommendations found in Amber's [MCPB.py #20](http://ambermd.org/tutorials/advanced/tutorial20/mcpbpy.htm) and [Heme](http://ambermd.org/tutorials/advanced/tutorial20/mcpbpy_heme.htm) tutorials.

# Installation

The script itself does not need installation, but it does require some stuff to be installed in your machine:

- [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/) + [pychimera](https://pychimera.readthedocs.io)
- [AmberTools](http://ambermd.org/AmberTools.php)
- Scipy

UCSF Chimera must be [downloaded and installed](https://www.cgl.ucsf.edu/chimera/download.html) manually. Everything else can be installed in a new [`conda`](https://docs.conda.io/en/latest/miniconda.html) environment with:

```
conda create -n mcpb -c insilichem -c AmberMD pychimera ambertools=18 scipy
conda activate mcpb
```

It has only been tested in Linux (Ubuntu 18.04).

# Examples

Right now, I have checked it against PDB IDs `1OKL` and `4ZF6`. In other words, [download a copy](https://github.com/jaimergp/prep-mcpb/raw/master/prep-mcpb.py) and run:


```
# Structure from tutorial 20
python prep-mcpb.py pdb:1OKL
```

```
# Structure from tutorial 20 (heme)
python prep-mcpb.py pdb:4ZF6
```

# Disclaimer and help 

This is a very rough attempt at familiarizing myself with the MCPB.py toolset. The main idea is to create a UCSF Chimera GUI extension to guide all the steps, but I will only do that after getting the basic workflow right. Please [submit an issue](https://github.com/jaimergp/prep-mcpb/issues) if you test it and find errors in any structure (attach the files if possible), or if you feel that the current approach does not fit your workflow.