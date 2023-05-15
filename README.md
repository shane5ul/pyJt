# pyJt

Extract continuous and discrete *retardation* spectra from creep compliance J(t). This program is based on papers for extracting the relaxation spectrum from G(t). The papers that describe the underlying method are:

+ Shanbhag, S., "pyReSpect: A Computer Program to Extract Discrete and Continuous Spectra from Stress Relaxation Experiments", Macromolecular Theory and Simulations, **2019**, 1900005 [doi: 10.1002/mats.201900005].
+ Takeh, A. and Shanbhag, S. "A computer program to extract the continuous and discrete relaxation spectra from dynamic viscoelastic measurements", Applied Rheology **2013**, 23, 24628. 

This work also uses Bayesian inference, which was subsequently documented for extracting the relaxation spectrum from the dynamic modulus $G^{*}(\omega)$.

+ Shanbhag, S., "Relaxation spectra using nonlinear Tikhonov regularization with a Bayesian criterion", Rheologica Acta, 2020, 59, 509 [doi: 10.1007/s00397-020-01212-w].

## Files

### Code Files

This repository contains two python modules `contSpec.py` `discSpec.py`. They extract the continuous and discrete retardation spectra from creep compliance data. (t versus J(t) experiment or simulation).

It containts a third module `common.py` which contains utilities required by both `contSpec.py` and `discSpec.py`.

### Input Files

The user is expected to supply two files:

+ `inp.dat` is used to control parameters and settings
+ `Jt.dat` (or similar) which contains two columns of data `t` and `J(t)`

### Output Files

Text files containting output from the code are stored in a directory `output/`. These include a fit of the data, the spectra, and other files relevant to the continuous or discrete spectra calculation. 

Graphical and onscreen output can be suppressed by appropriate flags in `inp.dat`.

### Status

Currently, this package has not been completely tested. I would appreciate learning about any issues that might arise. Currently this runs significantly slower than pyReSpect-time (a couple of minutes).

## Usage

Once `inp.dat` and `Jt.dat` are furnished, running the code is simple.

To get the continuous spectrum:

`python3 contSpec.py`

The **continuous spectrum must be extracted before the discrete spectrum** is computed. The discrete spectrum can then be calculated by

`python3 discSpec.py`


### Pre-requisites

The numbers in parenthesis show the version this has been tested on. 

python3 (3.6.9)
numpy (1.19.5)
scipy (1.5.4)
matplotlib (3.3.4)

## History

The code is based on the Matlab program [ReSpect](https://www.mathworks.com/matlabcentral/fileexchange/40458-respect), which extract the continuous and discrete relaxation spectra from frequency data, G*(w). Work was supported by National Science Foundation DMR grants number 0953002 and 1727870.


