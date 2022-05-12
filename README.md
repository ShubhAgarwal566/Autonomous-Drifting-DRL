## Create the conda environment

```conda env create -f environment.yml```

## Activate the environment
 
``` conda activate project_drift```

## Build Custom Gym Environment

```pip3 install --user -e gym/```

## To train the model

```python3 train_sac.py [--load true]```

## To test the model

```python3 test_sac.py```