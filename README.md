#  Setup
1. Install conda using [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Clone repository
```
git clone https://github.com/ChemRacer/VogLab_Book_Chapter.git
```
3. Install conda environment named voglab
```
cd VogLab_Book_Chapter/conda-envs
conda env create -f voglab.yml
```

4. Link conda environment to jupyter kernel
```
conda activate voglab
ipython kernel install --user --name=voglab
conda deactivate
```


