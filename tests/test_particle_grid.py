import pytest
import numpy as np


@pytest.fixture(scope='module', params=[123, 1234], ids=['N123','N1234'])
def N(request):
    return request.param


@pytest.fixture(scope='module', params=[1.0, 23.4], ids=['L1','L2'])
def box(request):
    return request.param


@pytest.fixture(scope='module', params=[123,456], ids=['seed1','seed2'])
def pos(request, N, box):
    seed = request.param

    # uniform random
    rng = np.random.default_rng(seed)
    pos = rng.random((N,3), dtype=np.float64)*box
    pos[pos >= box] = 0.

    return pos


@pytest.mark.parametrize('ngrid', [7, 16], ids=lambda v: f'ngrid{v}')
@pytest.mark.parametrize('nthread', [1, -1], ids=lambda v: f'nthread{v}')
def test_particle_grid_vs_numpy(pos, box, nthread, ngrid):
    '''Check our binning against numpy
    '''
    from numba_2pcf.particle_grid import particle_grid

    pgrid,offsets = particle_grid(pos, ngrid, box, nthread=nthread)

    # Sanity checks and boundary conditions
    assert offsets[0] == 0
    assert offsets[-1] == len(pos)
    assert np.diff(offsets).sum() == len(pos)
    assert (offsets >= 0).all()
    assert pgrid.shape == pos.shape
    assert np.isfinite(pgrid).all()

    grid_idx3 = (pos / (box/ngrid)).astype(np.int64)
    grid_idx1 = np.ravel_multi_index(grid_idx3.T, (ngrid,ngrid,ngrid))
    grid_iord = np.argsort(grid_idx1)
    ncell = ngrid**3

    np_pgrid = pos[grid_iord]
    np_offsets = np.empty(ncell+1, dtype=np.int64)
    np_offsets[0] = 0
    np_offsets[1:] = np.bincount(grid_idx1, minlength=ncell).cumsum()

    # do numpy and particle_grid give the same cell counts?
    assert np.all(np_offsets == offsets)

    # now, within each cell, make sure all the same particles are present
    for c in range(ncell):
        assert np.all(
                np.sort(pgrid[offsets[c]:offsets[c+1]], axis=0) ==
                np.sort(np_pgrid[np_offsets[c]:np_offsets[c+1]], axis=0)
                )
