'''
A simple 2PCF, using a particle grid and N^2
'''

import numpy as np
import numba as nb
import astropy.table
from astropy.table import Table

import chaining_mesh

__all__ = ['tpcf', 'jackknife']

_fastmath = True
_parallel = True

@nb.njit(fastmath=_fastmath)
def _1d_to_3d(i,ngrid):
    '''i is flat index, n1d is grid size'''
    
    X = np.empty(3,dtype=np.int64)
    X[0] = i // (ngrid[1]*ngrid[2])
    X[1] = i // ngrid[2] - X[0]*ngrid[2]
    X[2] = i % ngrid[2]
    
    return X

@nb.njit(fastmath=_fastmath)
def _do_cell_pair(pos1, pos2, Rmax, nbin, Xoff, counts):
    dtype = pos1.dtype
    inv_bw = dtype.type(nbin/Rmax)
    Rmax2 = Rmax*Rmax
    N1,N2 = len(pos1), len(pos2)
    for i in range(N1):
        p1 = pos1[i]
        for j in range(N2):
            p2 = pos2[j]
            # Early exit conditions
            # TODO: could exploit cell sorting better
            zdiff = np.abs(p1[2] - p2[2] + Xoff[2])
            if zdiff > Rmax:
                continue
            ydiff = np.abs(p1[1] - p2[1] + Xoff[1])
            if ydiff > Rmax:
                continue
            xdiff = np.abs(p1[0] - p2[0] + Xoff[0])
            if xdiff > Rmax:
                continue
            
            r2 = xdiff**2 + ydiff**2 + zdiff**2
            if r2 > Rmax2:
                continue
            r = np.sqrt(r2)
            b = int(r*inv_bw)
            counts[i,b] += 1
    

@nb.njit(parallel=_parallel,fastmath=_fastmath)
def _tpcf(psort, starts, ngrid, box, Rmax, nbin):
    dtype = psort.dtype
    
    ncell = np.prod(ngrid)
    st1d = starts  # ngrid^3 + 1
    
    nw = np.array([3,3,3])  # neighbor width
    nneigh = np.prod(nw)
    
    nthread = nb.get_num_threads()
    # mean
    thread_counts = np.zeros((nthread,nbin), dtype=np.int64)
    # covariance
    thread_cov = np.zeros((nthread,nbin*(nbin+1)//2), dtype=np.int64)
    
    for c in nb.prange(ncell):
        pn = st1d[c+1] - st1d[c]  # count in primary
        pcounts = np.zeros((pn,nbin),dtype=np.int64)  # per-particle bin counts
        
        for off in range(nneigh):
            t = nb.np.ufunc.parallel._get_thread_id()

            # global neighbor index
            c3d = _1d_to_3d(c,ngrid)
            off3d = _1d_to_3d(off,nw)
            d3d = c3d + off3d - 1

            # periodic neighbor index wrap
            Xoff = np.zeros(3, dtype=dtype)
            for j in range(3):
                if d3d[j] >= ngrid[j]:
                    d3d[j] -= ngrid[j]
                    Xoff[j] -= box
                if d3d[j] < 0:
                    d3d[j] += ngrid[j]
                    Xoff[j] += box
            # 1d neighbor index
            d = d3d[0]*ngrid[1]*ngrid[2] + d3d[1]*ngrid[2] + d3d[2]

            _do_cell_pair(psort[st1d[c]:st1d[c+1]],
                          psort[st1d[d]:st1d[d+1]],
                          Rmax, nbin, Xoff,
                          pcounts
                         )
        # no self-counts
        pcounts[:,0] -= 1
        
        # reduce per-particle counts into mean and cov
        thread_counts[t] += pcounts.sum(axis=0)
        for q in range(pn):
            ii = 0
            for i in range(nbin):
                for j in range(i,nbin):
                    thread_cov[t,ii] += pcounts[q,i]*pcounts[q,j]
                    ii += 1
    
    counts = thread_counts.sum(axis=0)
    cov = thread_cov.sum(axis=0)
    
    return counts, cov
    

def tpcf(pos, box, Rmax, nbin, nthread=-1, n1djack=None, cm_kwargs=None):
    if cm_kwargs is None:
        cm_kwargs = {}
    cm_kwargs = cm_kwargs.copy()
    if 'nthread' not in cm_kwargs:
        cm_kwargs['nthread'] = nthread
    if 'sort_in_cell' not in cm_kwargs:
        cm_kwargs['sort_in_cell'] = True
    
    if nthread == -1:
        nthread = nb.get_num_threads()
        
    # coerce inputs to match pos type
    box = pos.dtype.type(box)
    Rmax = pos.dtype.type(Rmax)
    
    ngrid = int(np.floor(box/Rmax))
    #ngrid = 3
    ngrid = (ngrid,)*3
    ngrid = np.atleast_1d(ngrid)
    
    psort, starts = chaining_mesh.chaining_mesh(pos, ngrid, box, **cm_kwargs)
    
    nb.set_num_threads(nthread)
    counts, covflat = _tpcf(psort, starts, ngrid, box, Rmax, nbin)
    
    # promote flattened cov to square
    cov = np.zeros((nbin,nbin),dtype=np.int64)
    cov[np.triu_indices(nbin)] = covflat
    cov = np.triu(cov) + np.tril(cov.T,-1)
    
    # compute xi from pairs
    N = len(pos)
    edges = np.linspace(0,Rmax,nbin+1)
    RR = np.diff(edges**3) * 4/3*np.pi * N*(N-1)/box**3
    xi = counts/RR - 1
    
    # compute cov from squared pair counts
    # note the numerator calculation uses integers to avoid catastrophic cancellation
    cov = (N*cov - counts.reshape(-1,1)*counts) / (N * RR.reshape(-1,1)*RR)
    
    # correlation matrix
    cor = cov / cov.diagonal()**0.5
    cor /= (cov.diagonal()**0.5).reshape(-1,1)
    
    t = Table(dict(rmin=edges[:-1],
                   rmax=edges[1:],
                   xi=xi,
                   npairs=counts,
                   cov=cov,
                   cor=cor,
                  ))
    t['rmid'] = (t['rmin'] + t['rmax'])/2.
    
    if n1djack:
        jack = jackknife(n1djack, pos, box, Rmax, nbin, nthread=nthread, cm_kwargs=cm_kwargs)
        for col in jack.colnames:
            if col in t.colnames:
                del jack[col]
        t = astropy.table.hstack((t,jack))
        
    return t

def jackknife(n1djack, pos, box, Rmax, nbin, nthread=-1, corrfunc=False, cm_kwargs=None):
    # use the chaining mesh to generate patches
    psort, offsets = chaining_mesh.chaining_mesh(pos, n1djack, box, nthread=nthread)
    del pos  # careful!
    occ = np.diff(offsets)
    
    all_res = []
    njack = n1djack**3
    for i in range(njack):
        pos_drop1 = np.empty((len(psort) - occ[i],3), dtype=psort.dtype)
        # copy all before the dropped patch, then all after
        pos_drop1[:offsets[i]] = psort[:offsets[i]]
        pos_drop1[offsets[i]:] = psort[offsets[i+1]:]
        
        if not corrfunc:
            res = tpcf(pos_drop1, box, Rmax, nbin, nthread=nthread, cm_kwargs=cm_kwargs)
        else:
            import Corrfunc.theory.DD
            bins = np.linspace(0,Rmax,nbin+1)
            res = Corrfunc.theory.DD(1, nthread, bins, *pos_drop1.T, boxsize=box, periodic=True)
            res = Table(res)
            res['npairs'][0] -= len(pos_drop1)
            res['xi'] = res['npairs']/(len(pos_drop1)/box**3 * (len(pos_drop1) - 1) * np.diff(bins**3) *4/3*np.pi) - 1
            res['rmid'] = (res['rmin'] + res['rmax'])/2.
        
        all_res += [res]
        
    jackres = all_res[0]['rmin','rmax','rmid'].copy()
    jackres['jack_xi'] = np.vstack([t['xi'] for t in all_res]).T
    jackres['jack_mean'] = jackres['jack_xi'].mean(axis=1)
    diff = jackres['jack_xi'] - jackres['jack_mean'].reshape(-1,1)
    jackres['jack_cov'] = (diff @ diff.T) * (njack - 1) / njack
    
    jackres['jack_cor'] = jackres['jack_cov'] / jackres['jack_cov'].diagonal()**0.5
    jackres['jack_cor'] /= (jackres['jack_cov'].diagonal()**0.5).reshape(-1,1)
    
    return jackres
