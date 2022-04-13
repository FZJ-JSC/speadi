from numba import njit, prange

from .histogram_distances import _compute_G_self, _compute_G_distinct
from .time_distance_matrix import _rtau_general_mic, _rtau_ortho_mic, _rtau_ortho_mic_self

opts = dict(parallel=True, fastmath=True, nogil=True, cache=False, debug=False)


@njit(['Tuple((f4[:,:,:],f4[:,:,:,:]))(f4[:,:,:],f4[:,:,:,:],f4[:,:,:],i4[:,:],i4[:,:],i4[:],i4[:],f4[:,:,:],f4[:],UniTuple(f4,2),i4,i4[:,:,:,:],f4[:])'], **opts)
def _append_Grts_general_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions, bin_edges):
    for i in prange(g1.shape[0]):
        for j in prange(g2.shape[0]):
            rt_self, rt_distinct = _rtau_general_mic(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            G_distinct[i, j] = _compute_G_distinct(rt_distinct, G_distinct[i, j], cuvol, bin_edges)
        G_self[i] += _compute_G_self(rt_self, G_self[i], cuvol, bin_edges)
    return G_self, G_distinct


@njit(['Tuple((f4[:,:,:],f4[:,:,:,:]))(f4[:,:,:],f4[:,:,:,:],f4[:,:,:],i4[:,:],i4[:,:],i4[:],i4[:],f4[:,:,:],f4[:],UniTuple(f4,2),i4,i4[:,:,:,:],f4[:])'], **opts)
def _append_Grts_ortho_mic(G_self, G_distinct, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins, unions, bin_edges):
    for i in prange(g1.shape[0]):
        for j in prange(g2.shape[0]):
            rt_self, rt_distinct = _rtau_ortho_mic(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            G_distinct[i, j] = _compute_G_distinct(rt_distinct, G_distinct[i, j], cuvol, bin_edges)
        G_self[i] = _compute_G_self(rt_self, G_self[i], cuvol, bin_edges)
    return G_self, G_distinct


@njit(['f4[:,:,:](f4[:,:,:],f4[:,:,:],i4[:,:],i4[:],f4[:,:,:],f4[:],UniTuple(f4,2),i4,i4[:,:,:,:],f4[:])'], **opts)
def _append_Grts_self(G_self, xyz, g1, g1_lens, cuvec, cuvol, r_range, nbins, unions, bin_edges):
    for i in prange(g1.shape[0]):
        rt_self = _rtau_ortho_mic_self(xyz, g1[i][:g1_lens[i]], cuvec)
        G_self[i] = _compute_G_self(rt_self, G_self[i], cuvol, bin_edges)
    return G_self
