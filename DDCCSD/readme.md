# Data-Driven Coupled-Cluster Singles and Doubles (DDCCSD)
This example uses the DDCCSD model to predict energies along the dissociation curve of a water molecule.

# Required python3 modules:
- numpy
- psi4
- os
- sklearn
- matplotlib

If you are missing any of these packages I would recommend using the following lines in your juypter notebook:
```
import sys
!{sys.executable} -m pip install missing_package_name 
```


# Files required to run DDCCSD_model.ipynb:
- XYZ coordinate files of traing set molecules : DDQC_Demo/DDCCSD/Water/Regular/Water5/ 
- XYZ coordinate files of test set molecules : DDQC_Demo/DDCCSD/Water/Water100/ 
- helper_CC_ML_old.py script: DDQC_Demo/DDCCSD/helper_CC_ML_old.py

# To run the DDCCSD_model.ipynb call jupyter in command line:
```
jupyter notebook DDCCSD_model.ipynb
```

