import yt
import os,sys
import random
from scipy.spatial import cKDTree
# from SPS_reader import SSP_interpolator
import numpy as np
import time as time
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.interpolate import interp1d
import cupy as cp
from cupyx.scipy.special import exp1

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

class Radiative_Transfer():
    def __init__(self,met,met_2,den,pos_0,dx,dt,t_now,stars,halo,timestep,pfs,opacity,lu,nu):
        freq_groups = np.array_split(np.arange(len(nu)),max(len(nu)/10000,1))
        self.spectra_final = None
        met_enter = ((met > 1e-5) | (met_2 > 1e-5))*(met >1e-6)
        if rank ==0:
            self.spectra_final = np.zeros((len(dx[met_enter]),len(nu)))
            np.maximum(opacity, 1e-100, out=opacity)
        for k,split_f in enumerate(freq_groups):
            spectra_accumulate = self.make_spectra(met,met_enter,den,pos_0,dx,dt,t_now,stars,halo,timestep,pfs,opacity[:, :, split_f],lu,nu[split_f])
            if rank==0:
                self.spectra_final[:, split_f] = spectra_accumulate

    def star_trajectory(self,stars,timestep,pfs,halo):
        starlist1 = stars[halo][timestep]
        starlist2 = stars[halo][timestep+1]
        id2_0 = starlist1['ids2']
        id2_1 = starlist2['ids2']
        common_ids, ind_0, ind_1 = np.intersect1d(
            id2_0, id2_1, return_indices=True)
        pos_0 = starlist1['positions2'][ind_0]
        pos_1 = starlist2['positions2'][ind_1]
        bool_in_2 = np.isin(id2_1,id2_0)
        t_0 = pfs[timestep][-1].astype(float)
        t_1 = pfs[timestep+1][-1].astype(float)
        bool_new = starlist2['age2'] < (t_1-t_0)/1e3
        id2_new = starlist2['ids2'][bool_new]
        ids_comb = np.append(common_ids,id2_new).astype(int)
        t_start =  t_0 - starlist1['age2'][ind_0]*1e3
        z_0 = pfs[timestep][-2].astype(float)
        a_0 = 1/(1+z_0)
        z_1 = pfs[timestep+1][-2].astype(float)
        a_1 = 1/(1+z_1)
        pos_born = starlist2['positions2'][bool_new] - \
                (t_1-t_0)*1e6*365.25*24*3600*starlist2['vels2'][bool_new]/\
                (starlist2['length_unit_pc']*3.08567758128e18)
        pos_0 = np.vstack((pos_0,pos_born))
        t_start = np.append(t_start,\
                            t_1-starlist2['age2'][bool_new]*1e3)
        pos_1 = np.vstack((pos_1,starlist2['positions2'][bool_new]))
        V = (pos_1-pos_0)/(t_1-t_0)
        sec_myr = 1e6*365.25*24*3600
        L_com = starlist2['length_unit_pc']*3.08567758128e18/a_1
        c_com = 2.99792458e10*sec_myr/L_com
        masses = np.append(starlist1['mass2'][ind_0],starlist2['mass2'][bool_new])
        metallicities = np.append(starlist1['met2'][ind_0],starlist2['met2'][bool_new])
        return t_0, t_start, V, pos_0, c_com ,t_1,a_0,a_1,masses,metallicities,L_com

    def find_emission(self,f_pos,t_now,t_0,t_1,V,c_com,t_start,pos_0,a_0,a_1,L_com):
        a_now = a_0+(a_1-a_0)*(t_now-t_0)/(t_1-t_0)
        if f_pos.ndim ==1:
            f_pos = f_pos[np.newaxis,:]
        pos_target = pos_0 + V*(t_now-t_0)
        R = f_pos - pos_target[:,np.newaxis]
        A = np.einsum('ij,ij->i',V,V) - (c_com/a_now)**2
        B = 2*np.einsum('ikj,ij->ki',R,V)
        C = np.einsum('ikj,ikj->ki',R,R)


        disc = B**2 - 4*A[np.newaxis,:]*C
        dt_light = (-B-np.sqrt(disc))/(2*A[np.newaxis,:])
        t_emit = t_now-dt_light
        pos_emit = pos_0 + V*(t_emit[:,:,np.newaxis]-t_0)
        return t_emit,pos_emit*L_com*a_now,a_now




    def job_organizer(self,root_ranks,job_i,Done,len_jobs,or_root=0,workers=3):
        root_now = -1
        rank_now = np.array([], dtype=int)
        time3 = None
        if rank in root_ranks and rank !=or_root:
            req = comm.isend(rank,tag=13,dest=or_root)
            req.wait()
            comm.Recv(Done,tag=20,source=or_root)
            if not Done[rank]:
                req = comm.irecv(tag=14,source=or_root)
                root_now = req.wait()
                req = comm.irecv(tag=17,source=or_root)
                job_i = req.wait()
                req = comm.irecv(tag=18,source=or_root)
                rank_now = req.wait()
        if rank not in root_ranks and rank !=or_root:
            req = comm.isend(rank,tag=12,dest=or_root)
            req.wait()
            comm.Recv(Done,tag=21,source=or_root)
            if not Done[rank]:
                time3 = time.time()
                req = comm.irecv(tag=15,source=or_root)
                rank_now = req.wait()
                req = comm.irecv(tag=16,source=or_root)
                job_i = req.wait()
                req = comm.irecv(tag=19,source=or_root)
                root_now = req.wait()
        if rank==or_root:
            root_or = np.append(root_ranks,or_root)
            bool_active = np.logical_not(np.isin(np.arange(nprocs),root_or))
            worker_ranks = np.arange(nprocs)[bool_active]
            if Done.sum() == len(Done)-1:
                Done[or_root] = True
            else:
                req = comm.irecv(tag=13,source=MPI.ANY_SOURCE)
                root_now = req.wait()
            if Done[bool_active].sum() == len(Done[bool_active]):
                Done[root_now] = True
            else:
                rank_now = np.array([])
                while len(rank_now) < workers and np.logical_not(Done[worker_ranks]).sum() > len(rank_now):
                    req = comm.irecv(tag=12,source=MPI.ANY_SOURCE)
                    rank_now = np.append(rank_now,req.wait()).astype(int)
            if job_i>=len_jobs:
                Done[rank_now] = True
                rank_now_i = -1*np.ones(workers)
                root_now_i = -1
            else:
                rank_now_i = rank_now
                root_now_i = root_now
            if not Done[or_root]:
                req = comm.Send((Done),tag=20,dest=root_now)
            if Done[root_ranks].sum() ==0:
                for rank_now_j in rank_now:
                    req = comm.Send((Done),tag=21,dest=rank_now_j)
            if not Done[or_root]:
                if Done[bool_active].sum() != len(Done[bool_active]) and np.all(~Done[rank_now]):
                    for rank_now_j in rank_now:
                        req = comm.isend(rank_now_i,tag=15,dest=rank_now_j)
                        req.wait()
                        req = comm.isend(job_i,tag=16,dest=rank_now_j)
                        req.wait()
                        req = comm.isend(root_now,tag=19,dest=rank_now_j)
                        req.wait()
                if not Done[root_now]:
                    req = comm.isend(root_now_i,tag=14,dest=root_now)
                    req.wait()
                    req = comm.isend(job_i,tag=17,dest=root_now)
                    req.wait()
                    req = comm.isend(rank_now,tag=18,dest=root_now)
                    req.wait()
            #print(rank_now,root_now,job_i,Done)
            job_i += 1
        return rank_now,root_now,job_i,Done,time3

    def job_organizer2(self,job_i,Done,len_jobs,worker_ranks,or_root=0):
        rank_now = -1
        if rank !=or_root:
            req = comm.isend(rank,tag=12,dest=or_root)
            req.wait()
            comm.Recv(Done,tag=20,source=or_root)
            if not Done[rank]:
                time3 = time.time()
                req = comm.irecv(tag=15,source=or_root)
                rank_now = req.wait()
                req = comm.irecv(tag=16,source=or_root)
                job_i = req.wait()
                #print(job_i)
        if rank==or_root:
            if Done.sum() == len(Done)-1:
                Done[or_root] = True
            else:
                req = comm.irecv(tag=12,source=MPI.ANY_SOURCE)
                rank_now = req.wait()
            if job_i>=len_jobs:
                Done[rank_now] = True
                rank_now_i = -1
            else:
                rank_now_i = rank_now
            if not Done[or_root]:
                req = comm.Send((Done),tag=20,dest=rank_now)
            #print(Done[bool_active].sum() != len(Done[bool_active]) and not Done[rank_now] and not Done[or_root])
            if not Done[or_root]:
                if Done[worker_ranks].sum() != len(Done[worker_ranks]) and not Done[rank_now]:
                    req = comm.isend(rank_now_i,tag=15,dest=rank_now)
                    req.wait()
                    req = comm.isend(job_i,tag=16,dest=rank_now)
                    req.wait()
            #print(rank_now,root_now,job_i,Done)
            rank_now = rank_now_i
            job_i += 1
        return rank_now,job_i,Done

    def job_scheduler_2(self,out_list,ranklim=1e99):
        ranks = np.arange(min(nprocs,ranklim)).astype(int)
        #print(ranks)
        jobs = {i.item(): [] for i in ranks}
        sto = {t: {} for t in out_list}
        return jobs, sto

    def gpu_ray_trace_4(self, ray_ind, ray_cell_local, dr, ipos, fpos, nu, den, mu, opacity, redshift,
                         target_cells, spectra_times, ray_hi, ray_lo, ray_w , source_lo, source_w, bool_past, gpu=0):
        """Computes spectral intensity for each target cell, summed over all sources.

            Parameters
            ----------
            ray_ind : np.ndarray
                Integer array of shape (n_rays, 3). Stores the indices of each ray segment's corresponding source, cell and endpoint.
                Calculated by ray_trace_1.
            
            ray_cell_local : np.ndarray
                Integer array of shape (n_rays, ). Local cell indices corresponding to the current memory chunk.

            dr : np.ndarray
                Double array of shape (n_rays, ). Length of each ray segment.
                Calculated by ray_trace_1.

            ipos : np.ndarray
                Double array of shape (n_ipos, 3). Position in simulation space of each ray start-point

            fpos : np.ndarray
                Double array of shape (n_fpos, 3). Position in simulation space of each ray end-point (cell center)

            nu : np.ndarray
                Double array of shape (n_nu, ). Each wavelength that tau is evaluated at.

            mu : np.ndarray
                Double array of shape (n_cells, ). Mean molecular weight across elements.

            opacity : np.ndarray
                Double array of shape (n_unique_cells, n_time_bins, n_nu). Stores previously computed absorption for all unique cells.

            redshift : np.ndarray
                Double array of shape (n_rays, ). Relative redshift for each ray segment.

            target_cells : np.ndarray
                Long array of shape (n_fpos, ). Target cell indices for this memory chunk.

            spectra_times : np.ndarray
                Double array of shape (n_time_bins, n_stars, n_nu). Stores historical relative spectra of each star.

            ray_hi : np.ndarray
                Integer array of shape (n_rays, ). Upper time-bin indices, calculated during ray intersection.
            
            ray_lo : np.ndarray
                Integer array of shape (n_rays, ). Lower time-bin indices, calculated during ray intersection.

            ray_w : np.ndarray
                Double array of shape (n_rays, ). Interpolation weights between time bins for ray segments.

            source_lo : np.ndarray
                Long array of shape (n_fpos, n_stars). Lower time-bin indices for each star at each cell, evaluated during emission.

            source_w : np.ndarray
                Double array of shape (n_fpos, n_stars). Interpolation weights between time bins for sources.
                   
            bool_past : np.ndarray
                Boolean array of shape (n_fpos, n_stars). Stores whether or not each star existed in the past time-step wrt. each cell
            
            


            Returns
            -------
            np.ndarray
                Double array of shape (n_fpos, n_nu). Integrated spectral intensity across all sources.
        """

        #Custom kernel to compute the interpolated and attenuated emission spectra 
        multiply_spectra = cp.ElementwiseKernel(
                'raw float64 spectra, raw int64 source_lo, raw float64 source_w, raw bool past, float64 attenuation, int32 n_stars, int32 n_nu',
                'float64 factor',
                '''
                int nu_i = i % n_nu;
                int star_i = (i / n_nu) % n_stars;
                int target_i = i / (n_stars*n_nu);
                int lo = source_lo[target_i*n_stars+star_i];
                double w = source_w[target_i*n_stars+star_i];
                int ind0 = (lo*n_stars+star_i)*n_nu+nu_i;
                int ind1 = ((lo+1)*n_stars+star_i)*n_nu+nu_i;
                factor = past[target_i*n_stars+star_i] ? attenuation*((1.0-w)*spectra[ind0]+w*spectra[ind1]) : 0.0;
                ''',
                'multiply_spectra')

        #raw-kernel implementation. Chunks freqs by block to reduce global memory loads, and enforces read-only memory on global arrays
        multiply_spectra_kernel = cp.RawKernel(r'''
                extern "C" __global__
                void multiply_spectra(
                    const double* __restrict__ spectra,      // [n_bins, n_stars, n_nu]
                    const unsigned int* __restrict__ source_lo,  // [n_fpos, n_stars]
                    const double* __restrict__ source_w,   // [n_fpos, n_stars]
                    const unsigned char* __restrict__ past,// [n_fpos, n_stars]
                    const double* __restrict__ attenuation,// [n_fpos, n_stars, n_nu]
                    double* __restrict__ factor,           // [n_fpos, n_stars, n_nu]
                    unsigned int n_stars,
                    unsigned int n_nu,
                    unsigned int n_fpos)
                {
                int pair = blockIdx.x;
                if(pair >= n_fpos * n_stars) return; //block out-of-bounds guard 
                __shared__ bool _past;
                __shared__ unsigned int _ind0;
                __shared__ unsigned int _ind1;
                __shared__ double _w;
                //first thread reads the common values into shared memory
                if(threadIdx.x == 0) {
                    _past = (past[pair] != 0);
                    if(_past) {
                        unsigned int star_idx = (unsigned int)pair % n_stars;
                        unsigned int lo = source_lo[pair];
                        _w = source_w[pair];
                        _ind0 = (lo * n_stars + star_idx) * n_nu;
                        _ind1 = ((lo+1) * n_stars + star_idx) * n_nu;
                    }
                }
                __syncthreads();
                //grid is arranged in (n_pairs, ceil(n_nu / threads_per_block)). 
                //the x axis is pairs. The y axis is frequency blocks. Each row corresponds to the
                //whole spectra for a given pair. Global memory reads are therefore
                //reduced by a factor of (n_nu / blockDim.y).

                //recover this thread's frequency:
                int nu_idx = threadIdx.x + blockIdx.y * blockDim.x;
                if(nu_idx >= n_nu) return; //thread out-of-bounds guard
                unsigned int global_out_idx = pair * n_nu + nu_idx;
                if(!_past) {
                    factor[global_out_idx] = 0.0;
                    return;
                }
                double spec_lo = spectra[_ind0 +nu_idx];
                double spec_hi = spectra[_ind1 +nu_idx];
                double atten = attenuation[global_out_idx];
                factor[global_out_idx] = atten * ((1.0 - _w)*spec_lo + (_w*spec_hi));
                }''', 'multiply_spectra')
        
        batch_interp_kernel = cp.RawKernel(r'''
                extern "C" __global__
                void batch_interp(const __restrict__ double* gnu,      // [n_nu]
                                const __restrict__ double* redshift, // [n_rays]
                                const __restrict__ double* chix,     // [n_unique_inds, n_nu]
                                const __restrict__ int* chi_ind,     // [n_rays]
                                const __restrict__ double* drt,      // [n_rays]
                                const __restrict__ int* i_s,         // [n_rays]
                                const __restrict__ int* j_s,         // [n_rays]
                                int n_nu,
                                int n_rays,
                                int n_fpos,
                                double *tau,           // [n_fpos, n_ipos, n_nu]
                                const unsigned char* is_target_cell,
                                double* tau_cell)

                {
                    int ray_idx = blockIdx.x;
                    int nu_idx = threadIdx.x + blockIdx.y * blockDim.x;
                    double eps = 1e-5;
                    if (ray_idx < n_rays && nu_idx < n_nu) {
                        double val = 0;
                        int row_offset = chi_ind[ray_idx] * n_nu;
                        if(redshift[ray_idx] > 1.0f - eps && redshift[ray_idx] < 1.0f + eps) {
                            val = chix[row_offset + nu_idx];
                        } else {
                            double red_gnu = gnu[nu_idx] / redshift[ray_idx];
                            int low = 0;
                            int mid = 0;
                            int high = n_nu - 1;
                            while (high - low > 1) {
                                mid = (low + high) >> 1;
                                bool go_right = (gnu[mid] < red_gnu);
                                low = go_right ? mid : low;
                                high = go_right ? high : mid;
                            }
                            if (low < 0) val = chix[row_offset];
                            else if (low >= n_nu) val = chix[row_offset + n_nu - 1];
                            else {
                                double x0 = gnu[low];
                                double x1 = gnu[low+1];
                                double y0 = chix[row_offset + low];
                                double y1 = chix[row_offset + low+1];
                                val = y0 + ((y1 - y0) / (x1 - x0)) * (red_gnu-x0);
                            }
                        }
                        int index = i_s[ray_idx] * n_fpos * n_nu + j_s[ray_idx] * n_nu + nu_idx;
                        double dtau = drt[ray_idx] * val;
                        atomicAdd(&tau[index], dtau);
                        if (is_target_cell[ray_idx]) atomicAdd(&tau_cell[index], dtau);
                    }
                }
                ''', 'batch_interp')
        with cp.cuda.Device(gpu):
            mH = 1.67e-24
            n_stars = ipos.shape[1]
            gtau_i_j = cp.zeros((len(fpos), n_stars, len(nu)), dtype=cp.float64)
            tau_cell = cp.zeros_like(gtau_i_j, dtype=cp.float64)
            gmu = cp.asarray(mu, dtype=cp.float64)
            gray_hi = cp.asarray(ray_hi, dtype=cp.int32)
            gray_lo = cp.asarray(ray_lo, dtype=cp.int32)
            gray_w = cp.asarray(ray_w, dtype=cp.float64)
            gopacity = cp.asarray(opacity, dtype=cp.float64)
            gnu = cp.asarray(nu, dtype=cp.float64)
            gredshift = cp.asarray(redshift, dtype=cp.float64)
            gray_ind = cp.asarray(ray_ind)
            gray_cell_local = cp.asarray(ray_cell_local, dtype=cp.int32)
            gdr = cp.asarray(dr, dtype=cp.float64)
            gtarget_cells = cp.asarray(target_cells, dtype=cp.int64)
            target_mask = gray_ind[:,1] == gtarget_cells[gray_ind[:,0]]
            is_target_cell = target_mask.astype(cp.uint8)
            dr_cell = cp.zeros((len(fpos),n_stars), dtype=cp.float64)
            dr_cell[gray_ind[target_mask,0],gray_ind[target_mask,2]] = gdr[target_mask]
            r_center = cp.linalg.norm(cp.asarray(fpos)[:,None,:]-cp.asarray(ipos),axis=2)
            r0 = r_center-dr_cell/2
            bool_inside_cell = r0 <0
            r0[bool_inside_cell] = dr_cell[bool_inside_cell]*1e-3
            dr_cell[bool_inside_cell] /= 2
            gdr[target_mask] =  dr_cell[gray_ind[target_mask,0],gray_ind[target_mask,2]]
            gden = cp.asarray(den, dtype=cp.float64)
            n_nu = len(gnu)
            n_fpos = gtau_i_j.shape[1]
            threads_per_block = 256
            ind_all = cp.arange(gopacity.shape[0], dtype=cp.int32)
            ray_ind_arange = cp.arange(len(ray_ind))
            split_inds = cp.array_split(ind_all, int(max(len(ind_all) / 200, 1)))
            bool_in_sum = cp.zeros(gray_ind.shape[0])
            for ind_i, inds in enumerate(split_inds):
                bool_in = cp.isin(gray_cell_local, inds)
                ray_ind_i = ray_ind_arange[bool_in]
                i_s, t_s, j_s = gray_ind[bool_in][:,0],gray_ind[bool_in][:,1],gray_ind[bool_in][:,2]
                gchix = gopacity[gray_cell_local[bool_in],gray_lo[bool_in],:]*(1-gray_w[bool_in,None]) +\
                        gopacity[gray_cell_local[bool_in],gray_hi[bool_in],:]*gray_w[bool_in,None]
                chix_i = gchix * (gden[t_s] / (mH * gmu[t_s]))[:,None]
                n_rays = len(i_s)
                drt = gdr[bool_in]
                blocks_per_ray = (n_nu + threads_per_block - 1) // threads_per_block
                grid = (n_rays, blocks_per_ray)
                block = (threads_per_block,)
                red = cp.ones_like(gredshift[ray_ind_i])
                #batch_interp_kernel expects int32
                i_s32 = i_s.astype(cp.int32)
                j_s32 = j_s.astype(cp.int32)
                chi_ind32 = cp.arange(n_rays, dtype=cp.int32)
                is_target_cell_i = is_target_cell[bool_in]
                batch_interp_kernel(
                    grid, block,
                    (gnu, red, chix_i, chi_ind32, drt, i_s32, j_s32,
                    n_nu, n_rays, n_fpos,is_target_cell_i, gtau_i_j,tau_cell))
                bool_in_sum += bool_in
            gtau_i_j -= tau_cell
            cp.exp(-gtau_i_j, out=gtau_i_j)
            ell = dr_cell[:,:,None]
            r0 = r0[:,:,None]
            tau_cell[:] = (1/r0-cp.exp(-tau_cell)/(r0+ell)+(tau_cell/ell)*cp.exp(tau_cell*r0/ell)*\
                (exp1(tau_cell*(r0+ell)/ell)-exp1(tau_cell*r0/ell)))/(4*cp.pi*ell)
            gtau_i_j *= tau_cell
            gspectra = cp.asarray(spectra_times, dtype=cp.float64)
            gpast = cp.asarray(bool_past, dtype=cp.uint8)
            gsource_lo = cp.asarray(source_lo, dtype=cp.uint32)
            gsource_w = cp.asarray(source_w, dtype=cp.float64)
            gfactor = cp.empty((n_fpos, n_stars, n_nu), dtype=cp.float64)
            threads_per_block = 256
            blocks_nu = (n_nu + threads_per_block - 1) // threads_per_block
            #attenuate & interpolate each ray
            grid = (n_fpos * n_stars, blocks_nu)
            block = (threads_per_block, 1)
            multiply_spectra_kernel(
                grid, block,
                (
                    gspectra,               # double*
                    gsource_lo,         # unsigned int*
                    gsource_w,              # double*
                    gpast,               # unsigned char*
                    gtau_i_j,           # double*
                    gfactor,                # double* 
                    np.uint32(n_stars),    # unsigned int
                    np.uint32(n_nu),       # unsigned int
                    np.uint32(n_fpos)      # unsigned int
                )
            )
            #multiply_spectra(gspectra, gsource_lo, gsource_w, gpast, gtau_i_j, n_stars, n_nu, gtau_i_j)

            #sum across all rays to get final spectra
            cell_intensity = gfactor.sum(axis=1)
            return cell_intensity.get()

    def gpu_ray_trace_1(self,ll, ur, dx, initial_pos, final_pos,gpu=0):
        use_hull=True
        bool_tmin_kernel = cp.ElementwiseKernel(
                'T tmax, T tmin',
                'bool bool_tmin',
                '''
                bool_tmin = (tmin < tmax) & (tmin < 1) & (tmax > 0);
                ''',
                'bool_tmin_kernel'
            )
        if use_hull:
            # bool_stars = (np.sum(initial_pos > self.star_center -0.3*self.halo_r,axis=1)==3) *\
            #             (np.sum(initial_pos < self.star_center +0.3*self.halo_r,axis=1)==3)
            #all_points = np.vstack((initial_pos[bool_stars],final_pos))
            all_points = np.vstack((initial_pos.reshape(-1, 3),final_pos))
            hull = ConvexHull(all_points)
            bool_bound = (self.contained(ll, hull, bigeps=dx[:, np.newaxis]) | self.contained(ur, hull, bigeps=dx[:, np.newaxis]) \
            | self.contained( (ur + ll) / 2,hull, bigeps=dx[:, np.newaxis]))
            # for istar in np.arange(len(initial_pos))[np.logical_not(bool_stars)]:
            #     all_points = np.vstack((initial_pos[istar],final_pos))
            #     hull = ConvexHull(all_points)
            #     bool_bound += contained(ll,hull,bigeps=dx[:,np.newaxis])+contained(ur,hull,bigeps=dx[:,np.newaxis])\
            #         +contained((ur+ll)/2,hull,bigeps=dx[:,np.newaxis])
            tot = max(initial_pos.shape[1]*len(final_pos)*bool_bound.sum()/4e8, 1)
            ll_max = int(max(bool_bound.sum()// (500/tot), 1))
            #print(bool_bound.sum()/len(ll),tot,ll_max)
        else:
            ll_max = max(ll.shape[0]//200, 1)
            bool_bound = np.arange(ll.shape[0])
        with cp.cuda.Device(gpu):
            ll_g = cp.asarray(ll, dtype=cp.float64)
            ur_g = cp.asarray(ur, dtype=cp.float64)
            ipos_g = cp.asarray(initial_pos, dtype=cp.float64)
            fpos_g = cp.asarray(final_pos, dtype=cp.float64)
            M = (cp.expand_dims(fpos_g, axis=1) - ipos_g)
            ll_ind = cp.arange(ll_g.shape[0])[bool_bound]
            split_ll_g = cp.array_split(ary=ll_ind, indices_or_sections=ll_max)
            tmin_f = cp.array([], dtype=cp.float64)
            tmax_f = cp.array([], dtype=cp.float64)
            ray_ind = cp.empty((0, 3), dtype=cp.int32)
            ray_ind_list = []
            tmin_list = []
            tmax_list = []
            for split_ll_i in split_ll_g:
                t0 = (ll_g[split_ll_i][None,:,None,:] - ipos_g[:,None,:,:]) / M[:,None,:,:]
                t1 = (ur_g[split_ll_i][None,:,None,:] - ipos_g[:,None,:,:]) / M[:,None,:,:]
                tmin = cp.minimum(t0, t1)
                tmax = cp.maximum(t0, t1)
                del t0, t1
                tmin = cp.max(tmin, axis=3)
                tmax = cp.min(tmax, axis=3)
                bool_tmin = bool_tmin_kernel(tmax, tmin)
                tmax_b = tmax[bool_tmin]
                del tmax
                tmin_b = tmin[bool_tmin]
                del tmin
                target_ind,cell_ind,star_ind = cp.where(bool_tmin)
                del bool_tmin
                ray_ind_g = cp.stack((target_ind, split_ll_i[cell_ind], star_ind), axis=1)
                del target_ind, cell_ind, star_ind
                tmin_list.append(tmin_b)
                tmax_list.append(tmax_b)
                ray_ind_list.append(ray_ind_g)
            ray_ind = cp.vstack(ray_ind_list)
            tmin_f = cp.concatenate(tmin_list)
            tmax_f = cp.concatenate(tmax_list)
            tmin_f_clamped = cp.maximum(tmin_f, 0)
            ray_fraction = 0.5 * (tmin_f_clamped + tmax_f)
            del tmin_f
            ray_ind_col2 = ray_ind[:,2]
            ray_ind_col0 = ray_ind[:,0]
            ipos_ray = ipos_g[ray_ind_col0, ray_ind_col2]
            p_close = tmin_f_clamped[:,None]*M[ray_ind_col0,ray_ind_col2] + ipos_ray
            p_far = tmax_f[:,None]*M[ray_ind_col0,ray_ind_col2] + ipos_ray
            dr = cp.linalg.norm(p_far-p_close, axis=1)
            return dr.get(), ray_ind.get(),ray_fraction.get()

    def contained(self,x,hull,bigeps=[]):
            eps = np.finfo(np.float32).eps
            A, b = hull.equations[:, :-1], hull.equations[:, -1:]
            # The hull is defined as all points x for which Ax + b <= 0.
            # We compare to a small positive value to account for floating
            # point issues.
            #
            # Assuming x is shape (m, d), output is boolean shape (m,).
            if len(bigeps)==0:
              return np.all(np.asarray(x) @ A.T + b.T < eps, axis=-1)
            else:
              return np.all(np.asarray(x) @ A.T + b.T < bigeps, axis=-1)

    def make_mu(self,met):
        Asplund = np.loadtxt('Asplund2021.txt',dtype=str)
        abuns = 10**(Asplund[:,2].astype(float))/1e12
        abuns = abuns[np.newaxis,:]*np.ones(len(met))[:,np.newaxis]
        abuns[:,2:] = abuns[:,2:]*met[:,np.newaxis]
        abuns /= abuns.sum(axis=1,keepdims=True)
        mu = np.zeros(len(met))
        for element in np.unique(Asplund[:,1].flatten()):
            element_ind = np.arange(len(Asplund[:,2]))[Asplund[:,1].flatten()==element]
            abun = abuns[:,element_ind].squeeze()
            mu += abun*periodictable.elements.symbol(element).mass
        return mu





    def make_spectra(self,met,met_enter,den,pos_0,dx,dt,t_now,stars,halo,timestep,pfs,opacity,lu,nu):
        pos_cm = pos_0 * lu
        dx_cm = np.asarray(dx)
        ll = pos_cm - dx_cm[:,None]/2
        ur = pos_cm + dx_cm[:,None]/2
        mu = self.make_mu(met)
        t_0, t_start, V, pos_s, c_com,t_1,\
            a_0,a_1,masses,metallicities,L_com = self.star_trajectory(stars,timestep,pfs,halo)
        n_light = 4
        dt_Myr = dt/(1e6*365.25*24*3600)
        n_history = n_light+1
        sim_files = pfs[:,0]
        ds = yt.load(sim_files[timestep])
        bool_born = t_start < t_now
        spectra_groups = np.array_split(np.arange(len(t_start))[bool_born],max(len(t_start[bool_born])/500,1))
        cell_groups = np.array_split(np.arange(len(pos_0[met_enter])),max(len(pos_0[met_enter])/50,1))
        freq_groups = np.array_split(np.arange(len(nu)),max(len(nu)/2000,1))
        jobs_0,sto = self.job_scheduler_2(np.arange(len(spectra_groups)))
        spectra_time_bin = np.linspace(t_now-dt_Myr*n_light, t_now,n_history)
        Done = np.full(nprocs,False)
        job_i = 0
        root_list = np.arange(1,len_gpu+1)
        dummy = 0
        for ranki in np.arange(nprocs):
            dummy = comm.bcast(dummy,root=ranki)
        job_i = 0
        spectra_accumulate = None
        for root_now_i in root_list:
            if rank ==0:
                opshape = np.array(opacity.shape).astype(int)
                comm.Send((opshape), dest=root_now_i, tag=100+root_now_i)
                comm.Send((opacity), dest=root_now_i, tag=200+root_now_i)
            if rank == root_now_i:
                opshape = np.zeros(3).astype(int)
                comm.Recv(opshape,tag=100+root_now_i,source=0)
                opacity = np.zeros(tuple(opshape,))
                comm.Recv(opacity,tag=200+root_now_i,source=0)
        if rank in root_list or rank ==0:
            spectra_accumulate = np.zeros((len(pos_0[met_enter]),len(nu)))
        while not Done[rank]:
            rank_now,root_now,job_i,Done,time3 = self.job_organizer(root_list,job_i,Done,len(sto),or_root=0)
            if rank in rank_now or rank==root_now:
                job_j = 0
                Done_2  = np.full(nprocs,True)
                Done_2[root_now] = False
                Done_2[rank_now] = False
                split = spectra_groups[job_i]
                V_i,t_start_i,pos_s_i = V[split],t_start[split],pos_s[split]
                lendr = len(t_start_i)
                jobs,sto = self.job_scheduler_2(np.arange(len(spectra_time_bin)))
                if rank == root_now:
                    spectra_times = np.zeros(((len(spectra_time_bin)),len(split),len(nu)))
                else:
                    spectra_times = {}
                while not Done_2[rank]:
                    rank_now_i,job_j,Done_2 = self.job_organizer2(job_j,Done_2,len(sto),rank_now,or_root=root_now)
                    if rank == rank_now_i:
                        ages = spectra_time_bin[job_j]-t_start[split]
                        lums, freq, spectra = SSP_interpolator(path_to_fsps,ds,path_to_fsps,ages,\
                                                               t_start_i,metallicities[split],\
                                                               masses[split,np.newaxis])
                        spectra_times[job_j] =  np.array([np.interp(nu, freq[::-1], spectra[i][::-1]) for i in range(len(spectra))])
                        jobs[rank_now_i].append(job_j)
                for rank_now_i in rank_now:
                    if rank==rank_now_i:
                        len_job_j = len(jobs[rank_now_i])
                        jobs[rank_now_i] = np.array(jobs[rank_now_i]).astype(int)
                        req = comm.isend(len_job_j,tag=7+rank_now_i,dest=root_now)
                        req.wait()
                        comm.Send((jobs[rank_now_i]), dest=root_now, tag=4)
                    elif rank==root_now:
                        req = comm.irecv(tag=7+rank_now_i,source=rank_now_i)
                        len_job_j = req.wait()
                        job_array = np.zeros(len_job_j).astype(int)
                        comm.Recv((job_array), tag=4,source=rank_now_i)
                        jobs[rank_now_i] = job_array
                for rank_now_i in rank_now:
                    for job_j in jobs[rank_now_i]:
                        if rank==rank_now_i:
                            comm.Send((spectra_times[job_j]), dest=root_now, tag=job_j)
                        elif rank==root_now:
                            spectra_times_i = np.zeros((len(split),len(nu)))
                            comm.Recv(spectra_times_i,tag=job_j,source=rank_now_i)
                            spectra_times[job_j] = spectra_times_i
                if rank==root_now:
                    for k,split_c in enumerate(cell_groups):
                        f_pos = pos_0[met_enter][split_c]
                        f_pos_cm = pos_cm[met_enter][split_c]
                        target_cells = np.flatnonzero(met_enter)[split_c]
                        t_emit_0, star_pos,a_now  = self.find_emission(f_pos,t_now,t_0,t_1,V_i,c_com,t_start_i,pos_s_i,a_0,a_1,L_com)
                        source_hi = np.clip(np.searchsorted(spectra_time_bin, t_emit_0), 1, len(spectra_time_bin)-1)
                        source_lo = source_hi-1
                        source_w = (t_emit_0-spectra_time_bin[source_lo])/(spectra_time_bin[source_hi]-spectra_time_bin[source_lo])
                        bool_past = t_emit_0 >= t_start_i[np.newaxis,:]
                        dr,ray_ind,r_frac = self.gpu_ray_trace_1(ll, ur, dx_cm, star_pos, f_pos_cm,gpu=root_now-1)
                        ray_emit_time = t_emit_0[ray_ind[:, 0],ray_ind[:, 2]]
                        ray_cross_time = ray_emit_time + ray_frac * (t_now - ray_emit_time)
                        ray_hi = np.clip(np.searchsorted(spectra_time_bin, ray_cross_time), 1, len(spectra_time_bin)-1)
                        ray_lo = ray_hi-1
                        ray_w = (ray_cross_time-spectra_time_bin[ray_lo])/(spectra_time_bin[ray_hi]-spectra_time_bin[ray_lo])
                        redshift = np.ones(len(ray_ind))
                        I_factor = np.zeros((len(split_c),len(nu)))
                        opacity_cells, ray_cell_local = np.unique(ray_ind[:, 1],return_inverse=True)
                        for k,split_f in enumerate(freq_groups):
                            opacity_i = opacity[opacity_cells][:, :, split_f]
                            I_factor[:,split_f]= self.gpu_ray_trace_4(ray_ind, ray_cell_local, dr, star_pos, f_pos_cm, nu[split_f], den, mu, opacity_i,\
                                redshift, target_cells, spectra_times[:,:,split_f], ray_hi, ray_lo, ray_w, \
                                source_lo,source_w,bool_past,gpu=root_now-1)
                        spectra_accumulate[split_c] += I_factor
                    jobs_0[root_now].append(job_i)
        for root_now_i in root_list:
            if rank==root_now_i:
                comm.Send((spectra_accumulate), dest=0, tag=30)
                spectra_accumulate = None
                opacity = None
            elif rank ==0:
                spectra_accumulate_i = np.zeros_like(spectra_accumulate)
                comm.Recv(spectra_accumulate_i,tag=30,source=rank_now_i)
                spectra_accumulate += spectra_accumulate_i
                spectra_accumulate_i = None
        return spectra_accumulate

def make_diffusion_kernel(halo,timestep,halo_tree,pfs,start=False):
    electrons = 0
    fracs = {}
    mH = 1.67e-24
    c_cgs = 29979245800
    sim_files = pfs[:,0]
    ds = yt.load(sim_files[timestep])
    center = halo_tree[halo][timestep]['Halo_Center']
    radius = halo_tree[halo][timestep]['Halo_Radius']*1.1
    reg = ds.sphere(center,radius)
    p_x,p_y,p_z = reg[('gas','x')].in_units('cm'),reg[('gas','y')].in_units('cm'),reg[('gas','z')].in_units('cm')
    pos = np.column_stack((p_x,p_y,p_z))
    lu = ds.length_unit.in_units('cm')
    bool_in = np.linalg.norm(pos-center*lu,axis=1) < halo_tree[halo][timestep]['Halo_Radius']*lu
    v_x,v_y,v_z = reg[('gas','velocity_x')].in_units('cm/s'),reg[('gas','velocity_y')].in_units('cm/s'),reg[('gas','velocity_z')].in_units('cm/s')
    den = reg[('gas','density')].in_units('g/cm**3').v
    met = reg[('metallicity')].in_units('Z_sun').v
    T = reg[('temperature')].in_units('K').v
    cell_volume = reg[('gas','volume')].v
    Asplund = np.loadtxt('Asplund2021.txt',dtype=str)
    abuns = 10**(Asplund[:,2].astype(float))/1e12
    abuns = abuns[np.newaxis,:]*np.ones(len(met))[:,np.newaxis]
    abuns[:,2:] = abuns[:,2:]*met[:,np.newaxis]
    abuns /= abuns.sum(axis=1,keepdims=True)
    v = np.column_stack((v_x,v_y,v_z))
    ids = np.arange(len(v))
    vnorm = np.linalg.norm(v,axis=1).max().v
    dx = reg['dx'].in_units('cm')
    dt = min(min(5*dx.min().v/(vnorm),1e5*24*365.25*3600),\
                 halo_tree[halo][timestep]['Halo_Radius']*0.5*lu.v/c_cgs)
    advected_pos = pos.v + v.v * dt
    tree = cKDTree(pos.v)
    dr, closein = tree.query(advected_pos,k=50,workers=-1)
    closein = closein.astype(np.int32, copy=False)
    shear = reg[("gas", "shear")].in_units("1/s")
    C_s = 0.1
    D = C_s**2 * dx.v**2 * shear.v
    if D.ndim == 0:
        variance = 4.0 * D * dt
    else:
        variance = 4.0 * D[:, None] * dt
    weights = -(dr**2) / variance
    weights += np.log(cell_volume[closein])
    weights -= weights.max(axis=1, keepdims=True)
    weights = np.exp(weights)
    weights /= weights.sum(axis=1, keepdims=True)

    if not start:
        return met, T, den, weights,cell_volume,bool_in,closein,dt,pos/ds.length_unit.in_units('cm'),lu
    else:
        h, h2, = reg[('gas', 'H_p0_number_density')],reg[('gas', 'H2_p0_number_density')]
        hm,h2p,he = reg[('gas', 'H_m1_number_density')],reg[('gas', 'H2_p1_number_density')],reg[('gas', 'He_p0_number_density')]
        hep,he2p,hp = reg[('gas', 'He_p1_number_density')],reg[('gas', 'He_p2_number_density')],reg[('gas', 'H_p1_number_density')]
        toth = abuns[:,0]/(h+2*h2+2*h2p+hm+hp)
        tothe = abuns[:,1]/(he+hep+he2p)
        h_num = h+2*h2+2*h2p+hm+hp
        fe = reg[('gas', 'El_number_density')]*toth
        return met, T, den,fe.v, weights,cell_volume,bool_in,closein,dt,\
            h*toth,hp*toth, h2*toth, hm*toth,h2p*toth,he*tothe,hep*tothe,he2p*tothe,h_num.v,\
            pos/lu,lu,dx

def find_next_condition(pos_0,halo,timestep,halo_tree,pfs):
    sim_files = pfs[:,0]
    ds = yt.load(sim_files[timestep])
    center = halo_tree[halo][timestep]['Halo_Center']
    radius = halo_tree[halo][timestep]['Halo_Radius']*2
    reg = ds.sphere(center,radius)
    p_x,p_y,p_z = reg[('gas','x')].in_units('cm'),reg[('gas','y')].in_units('cm'),reg[('gas','z')].in_units('cm')
    pos = np.column_stack((p_x,p_y,p_z))
    ll = pos-reg[('gas','dx')].in_units('cm')[:,np.newaxis]/2
    ur = pos+reg[('gas','dx')].in_units('cm')[:,np.newaxis]/2
    lu = ds.length_unit.in_units('cm')
    ll,ur = ll/lu,ur/lu
    v_x,v_y,v_z = reg[('gas','velocity_x')].in_units('cm/s'),reg[('gas','velocity_y')].in_units('cm/s'),reg[('gas','velocity_z')].in_units('cm/s')
    den = reg[('gas','density')].in_units('g/cm**3').v
    met = reg[('metallicity')].in_units('Z_sun').v
    T = reg[('temperature')].in_units('K').v
    cell_volume = reg[('gas','volume')].v
    h, h2, = reg[('gas', 'H_p0_number_density')],reg[('gas', 'H2_p0_number_density')]
    hm,h2p,he = reg[('gas', 'H_m1_number_density')],reg[('gas', 'H2_p1_number_density')],reg[('gas', 'He_p0_number_density')]
    hep,he2p,hp = reg[('gas', 'He_p1_number_density')],reg[('gas', 'He_p2_number_density')],reg[('gas', 'H_p1_number_density')]
    Asplund = np.loadtxt('Asplund2021.txt',dtype=str)
    abuns = 10**(Asplund[:,2].astype(float))/1e12
    abuns = abuns[np.newaxis,:]*np.ones(len(met))[:,np.newaxis]
    abuns[:,2:] = abuns[:,2:]*met[:,np.newaxis]
    abuns /= abuns.sum(axis=1,keepdims=True)
    toth = abuns[:,0]/(h+2*h2+2*h2p+hm+hp)
    tothe = abuns[:,1]/(he+hep+he2p)
    h_num = h+2*h2+2*h2p+hm+hp
    fe = reg[('gas', 'El_number_density')]*toth
    Tl = np.zeros(len(pos_0))
    metl = np.zeros(len(pos_0))
    denl = np.zeros(len(pos_0))
    hl = np.zeros(len(pos_0))
    hpl = np.zeros(len(pos_0))
    h2l = np.zeros(len(pos_0))
    hml = np.zeros(len(pos_0))
    h2pl = np.zeros(len(pos_0))
    hel = np.zeros(len(pos_0))
    hepl = np.zeros(len(pos_0))
    he2pl = np.zeros(len(pos_0))
    fel = np.zeros(len(pos_0))
    for i in range(len(pos_0)):
        bool_in = (
            (np.sum(pos_0[i] > ll, axis=1) == 3)
            * (np.sum(pos_0[i] <= ur, axis=1) == 3))
        volume = cell_volume[bool_in]
        volume_sum = volume.sum()
        Tl[i] = np.sum(T[bool_in] * volume) / volume_sum
        metl[i] = np.sum(met[bool_in] * volume) / volume_sum
        denl[i] = np.sum(den[bool_in] * volume) / volume_sum
        hl[i] = np.sum(h[bool_in] * volume) / volume_sum
        hpl[i] = np.sum(hp[bool_in] * volume) / volume_sum
        h2l[i] = np.sum(h2[bool_in] * volume) / volume_sum
        hml[i] = np.sum(hm[bool_in] * volume) / volume_sum
        h2pl[i] = np.sum(h2p[bool_in] * volume) / volume_sum
        hel[i] = np.sum(he[bool_in] * volume) / volume_sum
        hepl[i] = np.sum(hep[bool_in] * volume) / volume_sum
        he2pl[i] = np.sum(he2p[bool_in] * volume) / volume_sum
        fel[i] = np.sum(fe[bool_in] * volume) / volume_sum
    atoms = [hl, hpl, h2l, hml, h2pl, hel, hepl, he2pl]
    return Tl,metl,denl,pos/lu,atoms,fel


#EXPOSED FOR TESTING

def gpu_ray_trace_4(ray_ind, ray_cell_local, dr, ipos, fpos, nu, den, mu, opacity, redshift,
                         target_cells, spectra_times, ray_hi, ray_lo, ray_w , source_lo, source_w, bool_past, gpu=0):
        """Computes spectral intensity for each target cell, summed over all sources.

            Parameters
            ----------
            ray_ind : np.ndarray
                Integer array of shape (n_rays, 3). Stores the indices of each ray segment's corresponding source, cell and endpoint.
                Calculated by ray_trace_1.
            
            ray_cell_local : np.ndarray
                Integer array of shape (n_rays, ). Local cell indices corresponding to the current memory chunk.

            dr : np.ndarray
                Double array of shape (n_rays, ). Length of each ray segment.
                Calculated by ray_trace_1.

            ipos : np.ndarray
                Double array of shape (n_ipos, 3). Position in simulation space of each ray start-point

            fpos : np.ndarray
                Double array of shape (n_fpos, 3). Position in simulation space of each ray end-point (cell center)

            nu : np.ndarray
                Double array of shape (n_nu, ). Each wavelength that tau is evaluated at.

            mu : np.ndarray
                Double array of shape (n_cells, ). Mean molecular weight across elements.

            opacity : np.ndarray
                Double array of shape (n_unique_cells, n_time_bins, n_nu). Stores previously computed absorption for all unique cells.

            redshift : np.ndarray
                Double array of shape (n_rays, ). Relative redshift for each ray segment.

            target_cells : np.ndarray
                Long array of shape (n_fpos, ). Target cell indices for this memory chunk.

            spectra_times : np.ndarray
                Double array of shape (n_time_bins, n_stars, n_nu). Stores historical relative spectra of each star.

            ray_hi : np.ndarray
                Integer array of shape (n_rays, ). Upper time-bin indices, calculated during ray intersection.
            
            ray_lo : np.ndarray
                Integer array of shape (n_rays, ). Lower time-bin indices, calculated during ray intersection.

            ray_w : np.ndarray
                Double array of shape (n_rays, ). Interpolation weights between time bins for ray segments.

            source_lo : np.ndarray
                Long array of shape (n_fpos, n_stars). Lower time-bin indices for each star at each cell, evaluated during emission.

            source_w : np.ndarray
                Double array of shape (n_fpos, n_stars). Interpolation weights between time bins for sources.
                   
            bool_past : np.ndarray
                Boolean array of shape (n_fpos, n_stars). Stores whether or not each star existed in the past time-step wrt. each cell
            
            


            Returns
            -------
            np.ndarray
                Double array of shape (n_fpos, n_nu). Integrated spectral intensity across all sources.
        """

        #Custom kernel to compute the interpolated and attenuated emission spectra 
        multiply_spectra = cp.ElementwiseKernel(
                'raw float64 spectra, raw int64 source_lo, raw float64 source_w, raw bool past, float64 attenuation, int32 n_stars, int32 n_nu',
                'float64 factor',
                '''
                int nu_i = i % n_nu;
                int star_i = (i / n_nu) % n_stars;
                int target_i = i / (n_stars*n_nu);
                int lo = source_lo[target_i*n_stars+star_i];
                double w = source_w[target_i*n_stars+star_i];
                int ind0 = (lo*n_stars+star_i)*n_nu+nu_i;
                int ind1 = ((lo+1)*n_stars+star_i)*n_nu+nu_i;
                factor = past[target_i*n_stars+star_i] ? attenuation*((1.0-w)*spectra[ind0]+w*spectra[ind1]) : 0.0;
                ''',
                'multiply_spectra')

        #raw-kernel implementation. Chunks freqs by block to reduce global memory loads, and enforces read-only memory on global arrays
        multiply_spectra_kernel = cp.RawKernel(r'''
                extern "C" __global__
                void multiply_spectra(
                    const double* __restrict__ spectra,      // [n_bins, n_stars, n_nu]
                    const unsigned int* __restrict__ source_lo,  // [n_fpos, n_stars]
                    const double* __restrict__ source_w,   // [n_fpos, n_stars]
                    const unsigned char* __restrict__ past,// [n_fpos, n_stars]
                    const double* __restrict__ attenuation,// [n_fpos, n_stars, n_nu]
                    double* __restrict__ factor,           // [n_fpos, n_stars, n_nu]
                    unsigned int n_stars,
                    unsigned int n_nu,
                    unsigned int n_fpos)
                {
                int pair = blockIdx.x;
                if(pair >= n_fpos * n_stars) return; //block out-of-bounds guard 
                __shared__ bool _past;
                __shared__ unsigned int _ind0;
                __shared__ unsigned int _ind1;
                __shared__ double _w;
                //first thread reads the common values into shared memory
                if(threadIdx.x == 0) {
                    _past = (past[pair] != 0);
                    if(_past) {
                        unsigned int star_idx = (unsigned int)pair % n_stars;
                        unsigned int lo = source_lo[pair];
                        _w = source_w[pair];
                        _ind0 = (lo * n_stars + star_idx) * n_nu;
                        _ind1 = ((lo+1) * n_stars + star_idx) * n_nu;
                    }
                }
                __syncthreads();
                //grid is arranged in (n_pairs, ceil(n_nu / threads_per_block)). 
                //the x axis is pairs. The y axis is frequency blocks. Each row corresponds to the
                //whole spectra for a given pair. Global memory reads are therefore
                //reduced by a factor of (n_nu / blockDim.y).

                //recover this thread's frequency:
                int nu_idx = threadIdx.x + blockIdx.y * blockDim.x;
                if(nu_idx >= n_nu) return; //thread out-of-bounds guard
                unsigned int global_out_idx = pair * n_nu + nu_idx;
                if(!_past) {
                    factor[global_out_idx] = 0.0;
                    return;
                }
                double spec_lo = spectra[_ind0 +nu_idx];
                double spec_hi = spectra[_ind1 +nu_idx];
                double atten = attenuation[global_out_idx];
                factor[global_out_idx] = atten * ((1.0 - _w)*spec_lo + (_w*spec_hi));
                }''', 'multiply_spectra')
        
        batch_interp_kernel = cp.RawKernel(r'''
                extern "C" __global__
                void batch_interp(const double* gnu,      // [n_nu]
                                const double* __restrict__ redshift, // [n_rays]
                                const double* __restrict__ chix,     // [n_unique_inds, n_nu]
                                const int* __restrict__ chi_ind,     // [n_rays]
                                const double* __restrict__ drt,      // [n_rays]
                                const int* __restrict__ i_s,         // [n_rays]
                                const int* __restrict__ j_s,         // [n_rays]
                                int n_nu, 
                                int n_rays,
                                int n_fpos,
                                double *tau,           // [n_fpos, n_ipos, n_nu]
                                const unsigned char* is_target_cell,
                                double* tau_cell)

                {
                    int ray_idx = blockIdx.x;
                    int nu_idx = threadIdx.x + blockIdx.y * blockDim.x;
                    double eps = 1e-8;
                    if (ray_idx < n_rays && nu_idx < n_nu) {
                        double val = 0;
                        int row_offset = chi_ind[ray_idx] * n_nu;
                        if(redshift[ray_idx] > 1.0f - eps && redshift[ray_idx] < 1.0f + eps) {
                            val = chix[row_offset + nu_idx];
                        } else {
                            double red_gnu = gnu[nu_idx] / redshift[ray_idx];
                            int low = 0;
                            int mid = 0;
                            int high = n_nu - 1;
                            while (high - low > 1) {
                                mid = (low + high) >> 1;
                                bool go_right = (gnu[mid] < red_gnu);
                                low = go_right ? mid : low;
                                high = go_right ? high : mid;
                            }
                            if (low < 0) val = chix[row_offset];
                            else if (low >= n_nu) val = chix[row_offset + n_nu - 1];
                            else {
                                double x0 = gnu[low];
                                double x1 = gnu[low+1];
                                double y0 = chix[row_offset + low];
                                double y1 = chix[row_offset + low+1];
                                val = y0 + ((y1 - y0) / (x1 - x0)) * (red_gnu-x0);
                            }
                        }
                        int index = i_s[ray_idx] * n_fpos * n_nu + j_s[ray_idx] * n_nu + nu_idx;
                        double dtau = drt[ray_idx] * val;
                        atomicAdd(&tau[index], dtau);
                        if (is_target_cell[ray_idx]) atomicAdd(&tau_cell[index], dtau);
                    }
                }
                ''', 'batch_interp')
        with cp.cuda.Device(gpu):
            mH = 1.67e-24
            n_stars = ipos.shape[1]
            gtau_i_j = cp.zeros((len(fpos), n_stars, len(nu)), dtype=cp.float64)
            tau_cell = cp.zeros_like(gtau_i_j, dtype=cp.float64)
            gmu = cp.asarray(mu, dtype=cp.float64)
            gray_hi = cp.asarray(ray_hi, dtype=cp.int32)
            gray_lo = cp.asarray(ray_lo, dtype=cp.int32)
            gray_w = cp.asarray(ray_w, dtype=cp.float64)
            gopacity = cp.asarray(opacity, dtype=cp.float64)
            gnu = cp.asarray(nu, dtype=cp.float64)
            gredshift = cp.asarray(redshift, dtype=cp.float64)
            gray_ind = cp.asarray(ray_ind)
            gray_cell_local = cp.asarray(ray_cell_local, dtype=cp.int32)
            gdr = cp.asarray(dr, dtype=cp.float64)
            gtarget_cells = cp.asarray(target_cells, dtype=cp.int64)
            target_mask = gray_ind[:,1] == gtarget_cells[gray_ind[:,0]]
            is_target_cell = target_mask.astype(cp.uint8)
            dr_cell = cp.zeros((len(fpos),n_stars), dtype=cp.float64)
            dr_cell[gray_ind[target_mask,0],gray_ind[target_mask,2]] = gdr[target_mask]
            r_center = cp.linalg.norm(cp.asarray(fpos)[:,None,:]-cp.asarray(ipos),axis=2)
            r0 = r_center-dr_cell/2
            bool_inside_cell = r0 <0
            r0[bool_inside_cell] = dr_cell[bool_inside_cell]*1e-3
            dr_cell[bool_inside_cell] /= 2
            gdr[target_mask] =  dr_cell[gray_ind[target_mask,0],gray_ind[target_mask,2]]
            gden = cp.asarray(den, dtype=cp.float64)
            n_nu = len(gnu)
            n_fpos = gtau_i_j.shape[1]
            threads_per_block = 256
            ind_all = cp.arange(gopacity.shape[0], dtype=cp.int32)
            ray_ind_arange = cp.arange(len(ray_ind))
            split_inds = cp.array_split(ind_all, int(max(len(ind_all) / 200, 1)))
            bool_in_sum = cp.zeros(gray_ind.shape[0])
            for ind_i, inds in enumerate(split_inds):
                bool_in = cp.isin(gray_cell_local, inds)
                ray_ind_i = ray_ind_arange[bool_in]
                i_s, t_s, j_s = gray_ind[bool_in][:,0],gray_ind[bool_in][:,1],gray_ind[bool_in][:,2]
                gchix = gopacity[gray_cell_local[bool_in],gray_lo[bool_in],:]*(1-gray_w[bool_in,None]) +\
                        gopacity[gray_cell_local[bool_in],gray_hi[bool_in],:]*gray_w[bool_in,None]
                chix_i = gchix * (gden[t_s] / (mH * gmu[t_s]))[:,None]
                n_rays = len(i_s)
                drt = gdr[bool_in]
                blocks_per_ray = (n_nu + threads_per_block - 1) // threads_per_block
                grid = (n_rays, blocks_per_ray)
                block = (threads_per_block,)
                red = cp.ones_like(gredshift[ray_ind_i])
                #batch_interp_kernel expects int32
                i_s32 = i_s.astype(cp.int32)
                j_s32 = j_s.astype(cp.int32)
                chi_ind32 = cp.arange(n_rays, dtype=cp.int32)
                is_target_cell_i = is_target_cell[bool_in]
                batch_interp_kernel(
                    grid, block,
                    (gnu, red, chix_i, chi_ind32, drt, i_s32, j_s32,
                    n_nu, n_rays, n_fpos, gtau_i_j, is_target_cell_i,tau_cell))
                bool_in_sum += bool_in
            gtau_i_j -= tau_cell
            cp.exp(-gtau_i_j, out=gtau_i_j)
            ell = dr_cell[:,:,None]
            r0 = r0[:,:,None]
            
            # 1. Identify zero-length segments to mask out later
            mask_zero_ell = (ell == 0.0)
            
            # 2. Create "safe" variables to prevent CuPy from eager-evaluating divisions by zero
            ell_safe = cp.where(mask_zero_ell, 1.0, ell)
            r0_safe = cp.where(r0 == 0.0, 1e-10, r0) # Prevents 1/0 if source is dead center
            tau_safe = cp.where(tau_cell == 0.0, 1.0, tau_cell) # Prevents 1/0 in asymp branch
            
            x0 = tau_cell * r0_safe / ell_safe
            x1 = tau_cell * (r0_safe + ell_safe) / ell_safe
            
            # 3. Define the safe regime
            safe_mask = x0 < 100.0 
            
            # 4. Sanitize inputs for the exact calculation so CuPy never computes inf * 0
            x0_safe = cp.where(safe_mask, x0, 1.0)
            x1_safe = cp.where(safe_mask, x1, 1.0)
            
            # Compute Exact Solution ONLY on sanitized arrays
            exact_num = 1/r0_safe - cp.exp(-tau_cell)/(r0_safe+ell_safe) + \
                        (tau_cell/ell_safe) * cp.exp(x0_safe) * (exp1(x1_safe) - exp1(x0_safe))
            exact_solution = exact_num / (4*cp.pi*ell_safe)
            
            # Compute fully simplified Asymptotic Solution (1/r0 physically cancels out)
            asymp_solution = 1 / (4 * cp.pi * tau_safe * r0_safe**2)
            
            # 5. Recombine based on the safe mask
            tau_cell_new = cp.where(safe_mask, exact_solution, asymp_solution)
            
            # 6. Apply physical logic: zero-length segments contribute zero attenuation
            tau_cell[:] = cp.where(mask_zero_ell, 0.0, tau_cell_new)
             

            # tau_cell[:] = (1/r0-cp.exp(-tau_cell)/(r0+ell)+(tau_cell/ell)*cp.exp(tau_cell*r0/ell)*\
            #     (exp1(tau_cell*(r0+ell)/ell)-exp1(tau_cell*r0/ell)))/(4*cp.pi*ell)
            
            gtau_i_j *= tau_cell
            gspectra = cp.asarray(spectra_times, dtype=cp.float64)
            gpast = cp.asarray(bool_past, dtype=cp.uint8)
            gsource_lo = cp.asarray(source_lo, dtype=cp.uint32)
            gsource_w = cp.asarray(source_w, dtype=cp.float64)
            gfactor = cp.empty((n_fpos, n_stars, n_nu), dtype=cp.float64)
            threads_per_block = 256
            blocks_nu = (n_nu + threads_per_block - 1) // threads_per_block
            #attenuate & interpolate each ray
            grid = (n_fpos * n_stars, blocks_nu)
            block = (threads_per_block, 1)
            multiply_spectra_kernel(
                grid, block,
                (
                    gspectra,               # double*
                    gsource_lo,         # unsigned int*
                    gsource_w,              # double*
                    gpast,               # unsigned char*
                    gtau_i_j,           # double*
                    gfactor,                # double* 
                    np.uint32(n_stars),    # unsigned int
                    np.uint32(n_nu),       # unsigned int
                    np.uint32(n_fpos)      # unsigned int
                )
            )
            #multiply_spectra(gspectra, gsource_lo, gsource_w, gpast, gtau_i_j, n_stars, n_nu, gtau_i_j)

            #sum across all rays to get final spectra
            cell_intensity = gfactor.sum(axis=1)
            return cell_intensity.get()

def contained(x,hull,bigeps=[]):
            eps = np.finfo(np.float32).eps
            A, b = hull.equations[:, :-1], hull.equations[:, -1:]
            # The hull is defined as all points x for which Ax + b <= 0.
            # We compare to a small positive value to account for floating
            # point issues.
            #
            # Assuming x is shape (m, d), output is boolean shape (m,).
            if len(bigeps)==0:
                return np.all(np.asarray(x) @ A.T + b.T < eps, axis=-1)
            else:
                return np.all(np.asarray(x) @ A.T + b.T < bigeps, axis=-1)

def gpu_ray_trace_1(ll, ur, dx, initial_pos, final_pos,gpu=0):
        use_hull=True
        bool_tmin_kernel = cp.ElementwiseKernel(
                'T tmax, T tmin',
                'bool bool_tmin',
                '''
                bool_tmin = (tmin < tmax) & (tmin < 1) & (tmax > 0);
                ''',
                'bool_tmin_kernel'
            )
        if use_hull:
            # bool_stars = (np.sum(initial_pos > self.star_center -0.3*self.halo_r,axis=1)==3) *\
            #             (np.sum(initial_pos < self.star_center +0.3*self.halo_r,axis=1)==3)
            #all_points = np.vstack((initial_pos[bool_stars],final_pos))
            all_points = np.vstack((initial_pos.reshape(-1, 3),final_pos))
            hull = ConvexHull(all_points)
            bool_bound = (contained(ll, hull, bigeps=dx[:, np.newaxis]) | contained(ur, hull, bigeps=dx[:, np.newaxis]) \
            | contained( (ur + ll) / 2,hull, bigeps=dx[:, np.newaxis]))
            # for istar in np.arange(len(initial_pos))[np.logical_not(bool_stars)]:
            #     all_points = np.vstack((initial_pos[istar],final_pos))
            #     hull = ConvexHull(all_points)
            #     bool_bound += contained(ll,hull,bigeps=dx[:,np.newaxis])+contained(ur,hull,bigeps=dx[:,np.newaxis])\
            #         +contained((ur+ll)/2,hull,bigeps=dx[:,np.newaxis])
            tot = max(initial_pos.shape[1]*len(final_pos)*bool_bound.sum()/4e8, 1)
            ll_max = int(max(bool_bound.sum()// (500/tot), 1))
            #print(bool_bound.sum()/len(ll),tot,ll_max)
        else:
            ll_max = max(ll.shape[0]//200, 1)
            bool_bound = np.arange(ll.shape[0])
        with cp.cuda.Device(gpu):
            ll_g = cp.asarray(ll, dtype=cp.float64)
            ur_g = cp.asarray(ur, dtype=cp.float64)
            ipos_g = cp.asarray(initial_pos, dtype=cp.float64)
            fpos_g = cp.asarray(final_pos, dtype=cp.float64)
            M = (cp.expand_dims(fpos_g, axis=1) - ipos_g)
            ll_ind = cp.arange(ll_g.shape[0])[bool_bound]
            split_ll_g = cp.array_split(ary=ll_ind, indices_or_sections=ll_max)
            tmin_f = cp.array([], dtype=cp.float64)
            tmax_f = cp.array([], dtype=cp.float64)
            ray_ind = cp.empty((0, 3), dtype=cp.int32)
            ray_ind_list = []
            tmin_list = []
            tmax_list = []
            for split_ll_i in split_ll_g:
                t0 = (ll_g[split_ll_i][None,:,None,:] - ipos_g[:,None,:,:]) / M[:,None,:,:]
                t1 = (ur_g[split_ll_i][None,:,None,:] - ipos_g[:,None,:,:]) / M[:,None,:,:]
                tmin = cp.minimum(t0, t1)
                tmax = cp.maximum(t0, t1)
                del t0, t1
                tmin = cp.max(tmin, axis=3)
                tmax = cp.min(tmax, axis=3)
                bool_tmin = bool_tmin_kernel(tmax, tmin)
                tmax_b = tmax[bool_tmin]
                del tmax
                tmin_b = tmin[bool_tmin]
                del tmin
                target_ind,cell_ind,star_ind = cp.where(bool_tmin)
                del bool_tmin
                ray_ind_g = cp.stack((target_ind, split_ll_i[cell_ind], star_ind), axis=1)
                del target_ind, cell_ind, star_ind
                tmin_list.append(tmin_b)
                tmax_list.append(tmax_b)
                ray_ind_list.append(ray_ind_g)
            ray_ind = cp.vstack(ray_ind_list)
            tmin_f = cp.concatenate(tmin_list)
            tmax_f = cp.concatenate(tmax_list)
            tmin_f_clamped = cp.maximum(tmin_f, 0)
            ray_fraction = 0.5 * (tmin_f_clamped + tmax_f)
            del tmin_f
            ray_ind_col2 = ray_ind[:,2]
            ray_ind_col0 = ray_ind[:,0]
            ipos_ray = ipos_g[ray_ind_col0, ray_ind_col2]
            p_close = tmin_f_clamped[:,None]*M[ray_ind_col0,ray_ind_col2] + ipos_ray
            p_far = tmax_f[:,None]*M[ray_ind_col0,ray_ind_col2] + ipos_ray
            dr = cp.linalg.norm(p_far-p_close, axis=1)
            return dr.get(), ray_ind.get(),ray_fraction.get()
