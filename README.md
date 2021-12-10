# Setup
1. Install conda using [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. You will need a valid install of Psi4 and Psi4NumPy to run the DDCCSD tutorials. Link for installation information [Psi4NumPy](https://github.com/psi4/psi4numpy)
3. Clone repository
```
git clone https://github.com/ChemRacer/VogLab_Book_Chapter.git
```
4. Install conda environment named voglab
```
cd VogLab_Book_Chapter/conda-envs
conda env create -f voglab.yml
```

5. Link conda environment to jupyter kernel
```
conda activate voglab
ipython kernel install --user --name=voglab
conda deactivate
```

# Case Studies
To run the DDCCSD tutorial:
```
cd VogLab_Book_Chapter/DDCCSD/
jupyter notebook DDCCSD_model.ipynb
```

To run the DDCASPT2 tutorial:
```
cd VogLab_Book_Chapter/DDCASPT2/
jupyter notebook gen_pair.ipynb 
```
