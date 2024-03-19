# TEMIP_sensitivity

This repository contains all data, scripts and libraries to reproduce the results from the article "Sensitivity analysis of inverted model parameters from transient electromagnetic measurements affected by induced polarization effects" by Lukas Aigner, Dieter Werthmüller and Adrian Flores Orozco (except for the QGIS map).

Function and classes needed to run the scripts are located in the library subfolder. Please find the scripts within the Figures folder.
The scripts are configured to add a relative import path to os.sys

For everything related to the sensitivity analysis, please use the salib_tem.yml file to build the corresponding environment, while all the other scripts can be run using the empypg environment.

This work would not have been possible withou many other open-source libraries, so please have also a look at the following repositories and consider citing the corresponding articles:

https://github.com/gimli-org/gimli

https://github.com/emsig/empymod

https://github.com/florian-wagner/four-phase-inversion

https://github.com/zperzan/pyDGSA

https://github.com/hadrienmichel/pyBEL1D

If you find this work useful and consider publishing related work, please consider citing our article:
https://doi.org/10.1016/j.jappgeo.2024.105334
