# Data-Driven Coupled-Cluster Singles and Doubles (DDCCSD)
This example uses DDCCSD model to predict energies along the dissociation curve of water molecule.
For this model training sets with different number of training set molecules can be used.

```
jupyter notebook DDCCSD_model.ipynb
```

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
# Files required to run gen_pair.ipynb:
- XYZ coordinate files of traing set molecules : provided in Water folder
- XYZ coordinate files of test set molecules : provided in Water folder (Water100)
- helper_CC_ML_old.py script: provided
```

# Additionally a notebook is provided to show how to extract features from a XYZ coordinate file
