import pytest
import numpy as np

from numba_2pcf.cf import numba_2pcf
import Corrfunc

import os
thread_max = len(os.sched_getaffinity(0))


@pytest.fixture(scope='module', params=[1000, 12345], ids=lambda v: f'N{v}')
def N(request):
    return request.param


@pytest.fixture(scope='module', params=[1.0, 23.4], ids=lambda v: f'L{v}')
def box(request):
    return request.param


@pytest.fixture(scope='module', params=[123,456], ids=['seed1','seed2'])
def pos(request, N, box):
    seed = request.param

    # uniform random
    rng = np.random.default_rng(seed)
    pos = rng.random((N,3), dtype=np.float32)*box

    return pos


@pytest.mark.parametrize('Rmax_ratio', [0.04, 0.07], ids=lambda v: f'R{v}')
@pytest.mark.parametrize('nbin', [8,51], ids=lambda v: f'nb{v}')
@pytest.mark.parametrize('nthread', [1, thread_max], ids=lambda v: f'nthread{v}')
def test_cf(pos, Rmax_ratio, box, nthread, nbin):
    Rmax = Rmax_ratio*box
    nbin = nbin
    bins = np.linspace(0.,Rmax,nbin+1)

    res = numba_2pcf(pos, box, Rmax, nbin, nthread=nthread)
    corrfunc_res = numba_2pcf(pos, box, Rmax, nbin, nthread=nthread, corrfunc=True)

    for col in res.colnames:
        assert np.allclose(res[col], corrfunc_res[col], equal_nan=True)
