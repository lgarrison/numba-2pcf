'''
A numba implementation of a particle grid decomposition,
also known as a chaining mesh. The primary application
is building algorithms, like a 2PCF calculation, that
require some spatial partitioning. One can find all of
a particle's neighbors by querying its parent cell and
all neighbor cells.

This numba implementation is parallelized and is pretty
fast, and probably "fast enough"; the particle grid build
won't be the bottleneck in most applications. But really
the motivation is to have a simple Python code that enables
rapid iteration.

Note that numba can also call C functions, so using this
as a top-level partitioner doesn't preclude low-level
optimization of the core computations.
'''

import numpy as np
import numba as nb

__all__ = ['particle_grid']

@nb.njit
def _hist_worker(counts, icell):
    counts[:] = 0
    for i in range(len(icell)):
        counts[icell[i]] += 1
        

@nb.njit(parallel=True)
def _hist(ncell, icell):
    nthread = nb.get_num_threads()
    pad = 8
    thread_counts = np.empty((nthread,ncell+pad), dtype=np.int64)
    
    N = len(icell)
    for t in nb.prange(nthread):
        i = (t//nthread)*N
        j = ((t+1)//nthread)*N
        _hist_worker(thread_counts[t], icell[i:j])
        
    counts = thread_counts[:,:ncell].sum(axis=0)
    
    return counts
    

@nb.njit(parallel=True,fastmath=True)
def _particle_grid(p, ngrid, L, nthread, sort=False):
    nb.set_num_threads(nthread)
    
    ncell = np.prod(ngrid)
    nx,ny,nz = ngrid
    starts = np.empty(ncell+1, dtype=np.int64)
    dtype = p.dtype.type
    
    N = len(p)
    icell = np.empty(N, dtype=np.int64)
    
    invdx = dtype(nx/L)
    invdy = dtype(ny/L)
    invdz = dtype(nz/L)
    for i in nb.prange(N):
        ix = np.int64(p[i,0]*invdx)
        iy = np.int64(p[i,1]*invdy)
        iz = np.int64(p[i,2]*invdz)
        
        icell[i] = ix*ny*nz + iy*nz + iz
        
    occupation = _hist(ncell, icell)
    
    starts[0] = 0
    starts[1:] = np.cumsum(occupation)
    
    # load balance
    # TODO: is this serial loop slow? a parallel version is possible too
    tstarts = np.empty(nthread+1, dtype=np.int64) 
    tstarts[0] = 0
    tnext = 1
    laststart = 0
    N_per_thread = N//nthread
    for i in range(ncell):
        if tnext >= nthread:
            break
        if starts[i] - laststart >= N_per_thread:
            tstarts[tnext] = i
            laststart = starts[i]
            tnext += 1
    tstarts[tnext:] = ncell
    
    #assert (tstarts >= 0).all()
    
    psort = np.empty_like(p)
    nwritten = np.empty(ncell, dtype=np.int32)
    for t in nb.prange(nthread):
        cstart = tstarts[t]
        cend = tstarts[t+1]
        nwritten[cstart:cend] = 0
        
        # Note this funny algorithm: each thread looks at all particles,
        # but only treats those in its thread domain
        for i in range(N):
            ic = icell[i]
            if ic < cstart or ic >= cend:
                continue
            for j in range(3):
                psort[starts[ic] + nwritten[ic],j] = p[i,j]
            nwritten[ic] += 1
            
    #assert (nwritten == occupation).all()
    
    if sort:
        for c in nb.prange(ncell):
            cp = psort[starts[c]:starts[c+1]]
            cp[:] = cp[cp[:,2].argsort()]
    
    return psort, starts
    

def particle_grid(p, ngrid, box, nthread=-1, sort_in_cell=False):
    '''
    Parameters
    ==========
    p: array of shape (N,3)
        the points to sort into the particle grid
    ngrid: int, 3-tuple of ints
        the mesh size
    box: float
        the box size
    sort_in_cell: bool
        Inside each particle grid cell, sort on the
        z dimension. Default: False.
        
    Returns
    =======
    psort: array of shape (N,3)
        The particles, sorted into particle grid order
    offsets: integer array of length prod(ngrid)+1
        The starting indices of the particle grid cells
    '''
    if type(ngrid) is int:
        ngrid = np.array([ngrid]*3)
    ngrid = np.atleast_1d(ngrid)
    
    if nthread == -1:
        nthread = nb.get_num_threads()
    #print(f'Using {nthread} threads')
    
    psort, offsets = _particle_grid(p, ngrid, box, nthread, sort=sort_in_cell)
    
    return psort, offsets
