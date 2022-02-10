# numba-2pcf

A Numba-based two-point correlation function calculator using a grid decomposition

## Installation

```python
python -m pip install numba-2pcf
```

## Branches
The `particle-jackknife` branch contains an implementation of an idea for computing
the xi(r) variance based on the variance of the per-particle xi(r) measurements.
It doesn't seem to be measuring the right thing, but the code is left for posterity.
