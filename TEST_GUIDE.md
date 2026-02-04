# Testing guide

## Via PYTHONPATH

```
cd breaking-the-chain-intervention
PYTHONPATH=$(pwd) pytest
```

## Via installing repo as a package in editable mode

1. Activate environment you work in, e.g.: 
```
conda activate break
```
(You can build conda environment using `break.yml` file)

2. Go to project directory:
```
cd breaking-the-chain-intervention
```

3. Ensure you use the right `pip` -- should be from the environment:
```
which pip
```

4. Install repository:
```
pip install -e .
```

5. Run all tests:
```
pytest
```

