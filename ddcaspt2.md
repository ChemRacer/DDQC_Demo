---
layout: page 
title: DDCASPT2 
category: DDCASPT2 
---

In this tutorial, the potential energies curve of the symmetric angle bend of ozone is shown using DDCASPT2. 


Data has been packaged into pickle or csv files for this method. 

The features can be opened:
```python
with open('feats.pickle', 'rb') as handle:
    X = pickle.load(handle)
```


The targets can be opened:
```python
with open('targets.pickle', 'rb') as handle:
    Y = pickle.load(handle)
```

The keys, containing the bond angles, can be opened:
```python
with open('keys.pickle', 'rb') as handle:
    keys = pickle.load(handle)
```

The CASPT2 energies for the full curve:
```python
caspt2=pd.read_csv('caspt2.csv',index_col='Label').drop(columns='Unnamed: 0')
```

The CASSCF energies for the full curve:
```python
casscf=pd.read_csv('casscf.csv',index_col='Label').drop(columns='Unnamed: 0').loc[map(float,test_ind)].rename(columns={'SCF':0})
```


The total correlation energies for the full curve:
```python
E1Dict=pd.read_csv("E2.csv").rename(columns={'Unnamed: 0':'Label'}).set_index('Label')
```

[[/images/PEC_4_3_1D_VDZP_LC.png]]
[[/images/scaled_new_4_3_1D_VDZP_pair.png]]
[[/images/scaled_new_4_3_1D_VDZP_LC.png]]

