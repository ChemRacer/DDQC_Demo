# Data-Driven Complete Active Space Second-Order Perturbation Theory (DDCASPT2)
This is an example of the DDCASPT2 method using ozone with a minimal active space of (4,3), where the 2pz orbitals of each oxygen atom are in the active space, and using an ANO-RCC-VDZP basis set.

Data runs from 110-170 degrees for a total of 121 points. To run the jupyter notebook:

```
jupyter notebook gen_pair.ipynb
```

# Required python3 modules:
- matplotlib
- numpy as np
- pandas
- seaborn
- os
- pickle
- sklearn
- mpl_toolkits

If you are missing any of these packages I would recommend using the following lines in your juypter notebook:
```
import sys
!{sys.executable} -m pip install missing_package_name 
```
# Files required to run gen_pair.ipynb:
- The angles (106-180 degrees): keys.pickle
- The regression targets Y (pair energies): targets.pickle
- The feature set, X: feats.pickle
- True CASSCF energies: casscf.csv
- True CASPT2 energies: caspt2.csv
- The true correlation energies (E2): E2.csv
