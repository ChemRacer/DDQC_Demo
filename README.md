# Data-Driven Quantum Chemistry Case Studies
This repository provides examples of data-driven quantum chemistry (DDQC) methods from the "Machine Learning for Accelerating and Improving *ab initio* Wave Function-based Methods" book chapter in  *Quantum Chemistry in the Age of Machine Learning*.

## Code Versions
- DDCCSD=0.1
- DDCASPT2=0.1


## Setup
1. Install conda using [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. You will need a valid install of Psi4 and Psi4NumPy to run the DDCCSD tutorials. Link for installation information [Psi4NumPy](https://github.com/psi4/psi4numpy)
3. Clone repository
```
git clone https://github.com/ChemRacer/DDQC_Demo.git
```
4. Install conda environment named ddqc_demo
```
cd DDQC_Demo/conda-envs
conda env create -f ddqc_demo.yml
```

5. Link conda environment to jupyter kernel
```
conda activate ddqc_demo
ipython kernel install --user --name=ddqc_demo
conda deactivate
```

## Case Studies
To run the DDCCSD tutorial:
```
cd DDQC_Demo/DDCCSD/
jupyter notebook DDCCSD_model.ipynb
```

To run the DDCASPT2 tutorial:
```
cd DDQC_Demo/DDCASPT2/
jupyter notebook gen_pair.ipynb
```
