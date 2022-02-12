# numba-2pcf

[![tests](https://github.com/lgarrison/numba-2pcf/actions/workflows/python.yml/badge.svg)](https://github.com/lgarrison/numba-2pcf/actions/workflows/test.yml)

A Numba-based two-point correlation function (2PCF) calculator using a grid decomposition.
Like [Corrfunc](https://github.com/manodeep/corrfunc), but written in [Numba](https://numba.pydata.org/),
with simplicity and hackability in mind.

## Installation

```console
$ git clone https://github.com/lgarrison/numba-2pcf.git
$ cd numba-2pcf
$ python -m pip install -e .
```

## Example
```python
from numba_2pcf.cf import numba_2pcf
import numpy as np

rng = np.random.default_rng(123)
N = 10**6
box = 2.
pos = rng.random((N,3), dtype=np.float32)*box

res = numba_2pcf(pos, box, Rmax=0.05, nbin=10)
res.pprint_all()
```

```
        rmin                 rmax                 rmid                    xi            npairs 
-------------------- -------------------- -------------------- ----------------------- --------
                 0.0 0.005000000074505806 0.002500000037252903   -0.004519257448573177    65154
0.005000000074505806 0.010000000149011612  0.00750000011175871   0.0020113763064291135   459070
0.010000000149011612  0.01500000022351742 0.012500000186264515    0.000984359247434119  1244770
 0.01500000022351742 0.020000000298023225 0.017500000260770324  -6.616896085054336e-06  2421626
0.020000000298023225  0.02500000037252903 0.022500000335276125  0.00019365366488166558  3993210
 0.02500000037252903  0.03000000044703484 0.027500000409781934   5.769329601057471e-05  5956274
 0.03000000044703484  0.03500000052154064 0.032500000484287736   0.0006815801672250821  8317788
 0.03500000052154064  0.04000000059604645 0.037500000558793545    2.04711840243732e-05 11061240
 0.04000000059604645  0.04500000067055226 0.042500000633299354   9.313641918828885e-05 14203926
 0.04500000067055226  0.05000000074505806  0.04750000070780516 -0.00011690771042793813 17734818
```

## Performance
The goal of this project is not to provide the absolute best performance that
given hardware can produce, but it *is* a goal to provide as good performance
as Numba will let us reach (while keeping the code readable). So we pay special
attention to things like `dtype` (use `float32` particle inputs when possible!),
parallelization, and some early-exit conditions (when we know a pair can't fall
in any bin).

As a demonstration that this code provides passably good performance,
here's a dummy test of 10<sup>7</sup> unclustered data points in a 2 Gpc/*h* box
(so number density 1.2e-3 *h*<sup>3</sup>/Mpc<sup>3</sup>), with Rmax=150 Mpc/h
and bin width of 1 Mpc/*h*:

```python
from numba_2pcf.cf import numba_2pcf
import numpy as np

rng = np.random.default_rng(123)
N = 10**7
box = 2000
pos = rng.random((N,3), dtype=np.float32)*box

%timeit numba_2pcf(pos, box, Rmax=150, nbin=150, corrfunc=False, nthread=24)  # 3.5 s
%timeit numba_2pcf(pos, box, Rmax=150, nbin=150, corrfunc=True, nthread=24)  # 1.3 s
```

So within a factor of 3 of Corrfunc, and we aren't even exploiting the
symmetry of the autocorrelation (i.e. we count every pair twice). Not bad!


## Modifying the Code
### Typical workflow
The code is laid out in two files: [`src/numba_2pcf/cf.py`](src/numba_2pcf/cf.py)
and [`src/numba_2pcf/particle_grid.py`](src/numba_2pcf/particle_grid.py).  As the
names suggest, `particle_grid.py` organizes the particles into cells, and `cf.py`
does something with those cells (in this case, compute the 2PCF).

We'll focus on `cf.py`, since most users will want to modify that.

There are three important functions in `cf.py`:
- `_do_cell_pair()` contains the core computation;
- `_2pcf()` contains the loop over cell pairs;
- `numba_2pcf()` is the main entry point.

If all you need is to add some new pair-wise statistic, then you'll want to modify
`_do_cell_pair()`.  You may need to add new argument(s) to take new inputs (like weights)
or outputs (like your new statistic).  This means you'll also need to modify the
calling function, `_2pcf()`, so it can pass the new args.  Follow the example of
`npairs`: make an array like
```python
thread_mystat = np.zeros((nthread,nbin), dtype=np.float64)
```
whose outer dimension is over threads, then have each thread `t` pass `thread_mystat[t]`
to `_do_cell_pair()`.  After the cell pairs are done, perform a reduction over threads,
which is just a sum in the case of pair counts:
```python
mystat = thread_mystat.sum(axis=0)
```

Then, all that remains is to return that new statistic to `numba_2pcf()` and add it
as a column in the Astropy Table passed back to the user.

### Debugging
One of the benefits of a Numba implementation is that you can always comment out
the `@numba.njit` decorators, and the Numba code will become plain Python code.
And it's a lot easier to debug plain Python than Numba!

Here are a few other debugging tips:
- Set `_parallel = False` at the top of `cf.py`.
- Call `numba_2pcf()` with `nthread=1`
- Make sure your modified code still gives the raw pair counts as Corrfunc (use `corrfunc=True` in `numba_2pcf()`)

### Testing Against Corrfunc
The code is [tested against Corrfunc](tests/test_cf.py). And actually, the
`numba_2pcf()` function takes a flag `corrfunc=True` that calls Corrfunc
instead of the Numba implementation to make such testing even easier.


## Algorithmic Details
`numba_2pcf` works a lot like Corrfunc, or any other grid-based 2PCF code: the
3D volume is divided into a grid of cells at least `Rmax` in size, where `Rmax`
is the maximum radius of the correlation function measurement. Then, we know
all valid particle pairs must be in neighboring cells. So the task is simply
to loop through each cell in the grid, pairing it with each of its 26 neighbors
(plus itself).  We parallelize over cell pairs, and add up all the pair counts
across threads at the end.

This grid decomposition prunes distant pairwise comparisons, so even though
the runtime still formally scales as O(N<sup>2</sup>), it makes the 2PCF
tractable for many realistic problems in cosmology and large-scale structure.

A Numba implementation isn't likely to beat Corrfunc on speed, but Numba
can still be fast enough to be useful (especially when the computation parallelizes
well).  The idea is that this code provides a "fast enough" parallel implementation
while still being highly readable—the 2PCF implementation is about 150 lines
of code, and the gridding scheme 100 lines.


## Branches
The `particle-jackknife` branch contains an implementation of an idea for computing
the xi(r) variance based on the variance of the per-particle xi(r) measurements.
It doesn't seem to be measuring the right thing, but the code is left for posterity.


## Acknowledgments
This repo was generated from [@DFM's Cookiecutter Template](https://github.com/dfm/cookiecutter-python). Thanks, DFM!
