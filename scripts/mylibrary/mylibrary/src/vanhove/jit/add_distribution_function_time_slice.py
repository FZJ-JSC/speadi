from numba import njit, prange

from mylibrary.src.vanhove.jit.histogram_distances import _compute_Grt
from mylibrary.src.vanhove.jit.time_distance_matrix import _compute_rt_mic, _compute_rt_mic_self


@njit(['f4[:,:,:,:](f4[:,:,:,:],f4[:,:,:],i8[:,:],i8[:,:],i8[:],i8[:],f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
def _append_Grts(G_rts, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins):
    for i in prange(g1.shape[0]):
        for j in range(g2.shape[0]):
            rt_array = _compute_rt_mic(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            G_rts[i, j] += _compute_Grt(rt_array, cuvol, r_range, nbins)
    return G_rts


@njit(['f4[:,:,:,:](f4[:,:,:,:],f4[:,:,:],i8[:,:],i8[:,:],i8[:],i8[:],f4[:,:,:],f4[:],UniTuple(f8,2),i8)'], parallel=True, fastmath=True, nogil=True)
def _append_Grts_self(G_rts, xyz, g1, g2, g1_lens, g2_lens, cuvec, cuvol, r_range, nbins):
    for i in prange(g1.shape[0]):
        for j in range(g2.shape[0]):
            rt_array = _compute_rt_mic_self(xyz, g1[i][:g1_lens[i]], g2[j][:g2_lens[j]], cuvec)
            G_rts[i, j] += _compute_Grt(rt_array, cuvol, r_range, nbins)
    return G_rts