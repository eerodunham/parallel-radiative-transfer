import yt
import numpy as np
import pandas as pd
import time
import cupy as cp
from cupy import fuse
import cupyx
from cupyx.profiler import benchmark, profile
from hyperion.dust import SphericalDust
import os
import matplotlib.pyplot as plt

os.environ['CUPY_ACCELERATORS'] = 'cub_python'

def cupy_ray_trace_1(ipos, fpos, ll, ur):
    '''define custom kernels'''
    bool_tmin_kernel = cp.ElementwiseKernel(
                'T tmax, T tmin',  
                'bool bool_tmin',        
                '''
                bool_tmin = (tmin < tmax) & (tmin < 1) & (tmax > 0);
                ''', 
                'bool_tmin_kernel'    
            )
    ll_g = cp.asarray(ll, dtype=cp.float64)
    ur_g = cp.asarray(ur, dtype=cp.float64)
    ipos_g = cp.asarray(ipos, dtype=cp.float64)
    fpos_g = cp.asarray(fpos, dtype=cp.float64)
    M = (cp.expand_dims(fpos_g, axis=1) - ipos_g)
    ll_ind = cp.arange(ll_g.shape[0])
    ll_max = max(ll_g.shape[0]//200, 1)
    split_ll_g = cp.array_split(ary=ll_ind, indices_or_sections=ll_max)
    ray_ind = cp.array([], dtype=cp.int16)
    ray_ind_list = []
    tmin_list = []
    tmax_list = []
    for split_ll_i in split_ll_g:
        t0 = (cp.expand_dims(ll_g[split_ll_i], axis=1) - ipos_g) / cp.expand_dims(M, axis=1) 
        t1 = (cp.expand_dims(ur_g[split_ll_i], axis=1) - ipos_g) / cp.expand_dims(M, axis=1) 
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
    del tmin_f
    ray_ind_col2 = ray_ind[:, 2]
    ray_ind_col0 = ray_ind[:,0]
    p_close = cp.expand_dims(tmin_f_clamped, axis=1)*M[ray_ind_col0,ray_ind_col2]+ipos_g[ray_ind_col2]
    p_far = cp.expand_dims(tmax_f, axis=1)*M[ray_ind_col0,ray_ind_col2]+ipos_g[ray_ind_col2]
    dr = cp.linalg.norm(p_far-p_close, axis=1)
    return dr.get(), ray_ind.get()


def cupy_ray_trace_4(ray_ind, dr, ipos, fpos, nu, svels, metals, den, i_temp,
                     chishe, chismet, chivhe, chivmet, redshift):
    '''define custom kernels'''
    chix_kernel = cp.ElementwiseKernel(
                'T Z, T chishe, T chismet, T chivhe, T chivmet, T den',
                'T chix',
                '''
                chix = (chishe + 0.0204*Z*chismet + chivhe + 0.0204*Z*chivmet) * den;
                ''',
                'chix_kernel'
            )
    batch_interp_kernel = cp.RawKernel(r'''
            extern "C" __global__
            void batch_interp(const double* gnu,      // [n_nu]
                            const double* redshift, // [n_rays]
                            const double* chix,     // [n_unique_inds, n_nu]
                            const int* chi_ind,     // [n_rays]
                            const double* drt,      // [n_rays]
                            const int* i_s,         // [n_rays]
                            const int* j_s,         // [n_rays]
                            int n_nu,
                            int n_rays,
                            int n_fpos,
                            double *tau)           // [n_fpos, n_ipos, n_nu]
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
                    atomicAdd(&tau[index], drt[ray_idx] * val);
                }
            }
            ''', 'batch_interp')
    gtau_i_j = cp.zeros((len(fpos), len(ipos), len(nu)), dtype=cp.float64)
    gredshift = cp.asarray(redshift(ipos, fpos, ray_ind, svels), dtype=cp.float64)
    gnu = cp.asarray(nu, dtype=cp.float64)
    gmetals = cp.asarray(metals, dtype=cp.float64)
    gi_temp = cp.asarray(i_temp)
    gray_ind = cp.asarray(ray_ind)
    gdr = cp.asarray(dr, dtype=cp.float64)
    gchishe = cp.asarray(chishe, dtype=cp.float64)
    gchismet = cp.asarray(chismet, dtype=cp.float64)
    gchivhe = cp.asarray(chivhe, dtype=cp.float64)
    gchivmet = cp.asarray(chivmet, dtype=cp.float64) 
    gden = cp.asarray(den, dtype=cp.float64)
    n_nu = len(gnu)
    n_fpos = gtau_i_j.shape[1]
    threads_per_block = 256
    ind_true = cp.arange(len(gmetals))
    ind_all = cp.unique(gray_ind[:,1][cp.isin(gray_ind[:, 1],ind_true)])
    ray_ind_arange = cp.arange(len(ray_ind))
    temp_ind = cp.unique(gi_temp[ind_all])
    print(len(ind_all))
    split_inds = cp.array_split(ind_all, int(max(len(ind_all) / 200, 1)))
    bool_in_sum = cp.zeros(gray_ind.shape[0])
    bigcount = 0
    for ind_i, inds in enumerate(split_inds):
        temp_j = cp.minimum(cp.searchsorted(temp_ind, gi_temp[inds]), len(temp_ind)-1)
        Z = cp.expand_dims(gmetals[inds], axis=1)
        chiden = (cp.expand_dims(gden[inds], axis=1) / mH)
        chix = chix_kernel(Z, gchishe[temp_j], gchismet[temp_j], gchivhe[temp_j], gchivmet[temp_j], chiden)
        # chix = gchivhe[temp_j] * chiden #uncomment if testing simplified version
        bool_in = cp.isin(gray_ind[:, 1], inds)
        ray_ind_i = ray_ind_arange[bool_in]
        i_s, t_s, j_s = gray_ind[bool_in][:,0],gray_ind[bool_in][:,1],gray_ind[bool_in][:,2]
        print(len(i_s), len(j_s), len(t_s))
        n_rays = len(i_s)
        chi_ind = cp.searchsorted(inds, t_s)
        drt = gdr[bool_in]
        blocks_per_ray = (n_nu + threads_per_block - 1) // threads_per_block
        grid = (n_rays, blocks_per_ray)
        block = (threads_per_block,)
        red = cp.ones_like(gredshift[ray_ind_i])
        #batch_interp_kernel expects int32
        chi_ind32 = chi_ind.astype(cp.int32)
        i_s32 = i_s.astype(cp.int32)
        j_s32 = j_s.astype(cp.int32)
        batch_interp_kernel(
            grid, block,
            (gnu, red, chix, chi_ind32, drt, i_s32, j_s32, 
            n_nu, n_rays, n_fpos, gtau_i_j)
        )
        bool_in_sum += bool_in
    bigcount = bool_in_sum.sum()
    cp.exp(-gtau_i_j, out=gtau_i_j)
    return gtau_i_j.get(), bigcount

class Halo:
    def __init__(self, timestep, testing=True):
        halo = np.load('halotree_2020_final.npy',allow_pickle=True).tolist()
        ds_path_1 = 'pfs_allsnaps_%s.txt' % halo_version
        file_list = np.loadtxt(ds_path_1,dtype=str)[:,0]
        ds = yt.load(file_list[timestep])
        plotd = np.load('plothype0_52.npy',allow_pickle=True).tolist()
        rad = halo['0'][5]['Halo_Radius']
        center = halo['0'][5]['Halo_Center']
        radius = ds.length_unit.in_units('cm') * rad
        reg = ds.box((center-rad),(center+rad))
        
        ll_x = ((reg['x']-reg['dx']/2)).in_units('cm').v
        ll_y = ((reg['y']-reg['dy']/2)).in_units('cm').v
        ll_z = ((reg['z']-reg['dz']/2)).in_units('cm').v
        self.ll = np.column_stack((np.column_stack((ll_x,ll_y)),ll_z))
        
        v_x = reg['velocity_x'].in_units('cm/s')
        v_y = reg['velocity_y'].in_units('cm/s')
        v_z = reg['velocity_z'].in_units('cm/s')
        v = np.concatenate((v_x,v_y,v_z))
        v = np.reshape(v,(len(v_x),3))
        self.vel = v.v
        self.dds = reg['dx'].in_units('cm').v
        self.ur = self.ll + self.dds[:, np.newaxis]
        stars = np.load('starlists_2020.npy',allow_pickle=True).tolist()
        print(stars['0'][0].keys())
        self.spos = stars['0'][timestep]['positions2'] * ds.length_unit.in_units("cm").v
        self.svels = stars['0'][timestep]['vels2']
        self.chivmet_0 = plotd['chivmet']
        self.chismet_0 = plotd['chismet']
        self.chivhe = plotd['chivhe']
        self.chishe = plotd['chishe']
        self.nu = plotd['nu']
        self.temps = reg['temperature'].in_units('K').v
        self.i_temp = np.minimum(np.searchsorted(plotd['temp'],self.temps),len(plotd['temp'])-1)
        self.metals = reg['metallicity'].in_units('Zsun').v
        self.ll_box = center-radius.v
        self.ur_box = center+radius.v
        self.den = np.array((reg['HI_Density']+reg['H2I_Density']+reg['H2II_Density']).in_units('g/cm**3').v)
        if(not testing):
            self.load_dust()
        else:
            self.load_mock_dust()

        # self.diff_nu_kernel =  cp.ReductionKernel(
        #     'T gnu_slice, raw T gnu',      
        #     'T out',          
        #     'abs((gnu[i+1] - gnu[i]) / gnu[i+1])', #Map function: get normalized diff
        #     'min(a, b)',      #Reduction function: take minimum, recursively reduce
        #     'out = a',        #Post-map: identity
        #     '1.0e20',         
        #     'diff_nu_kernel' 
        # )
        self.bool_red_kernel = cp.ElementwiseKernel(
            'T gredshift, T diff_nu',  
            'bool bool_red_out',        
            '''
            bool_red_out = (abs(gredshift - 1) > 0.5 * diff_nu);
            ''', 
            'bool_red_kernel'        
        )
        self.bool_tmin_kernel = cp.ElementwiseKernel(
            'T tmax, T tmin',  
            'bool bool_tmin',        
            '''
            bool_tmin = (tmin < tmax) & (tmin < 1) & (tmax > 0);
            ''', 
            'bool_tmin_kernel'    
        )
        self.chix_kernel = cp.ElementwiseKernel(
            'T Z, T chishe, T chismet, T chivhe, T chivmet, T den',
            'T chix',
            '''
            chix = (chishe + 0.0204*Z*chismet + chivhe + 0.0204*Z*chivmet) * den;
            ''',
            'chix_kernel'
        )

        self.batch_interp_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void batch_interp(const double* gnu,      // [n_nu]
                        const double* redshift, // [n_rays]
                        const double* chix,     // [n_unique_inds, n_nu]
                        const int* chi_ind,     // [n_rays]
                        const double* drt,      // [n_rays]
                        const int* i_s,         // [n_rays]
                        const int* j_s,         // [n_rays]
                        int n_nu,
                        int n_rays,
                        int n_fpos,
                        double *tau)           // [n_fpos, n_ipos, n_nu]
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
                atomicAdd(&tau[index], drt[ray_idx] * val);
            }
        }
        ''', 'batch_interp')

        self.interp_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void interp(const double* gnu,      // [n_nu]
                    const double redshift, // [n_rays]
                    const double* chix,
                    int n_nu,
                    double* interp) 
        {
            int nu_idx = threadIdx.x + blockIdx.x * blockDim.x;
            double eps = 1e-5;
            if(nu_idx < n_nu) {
                double red_gnu = gnu[nu_idx] / redshift;
                int low = 0;
                int mid = 0;
                int high = n_nu - 1;
                while (high - low > 1) {
                    mid = (low + high) >> 1;
                    bool go_right = (gnu[mid] < red_gnu);
                    low = go_right ? mid : low;
                    high = go_right ? high : mid;
                }
                double x0 = gnu[low];
                double x1 = gnu[low+1];                
                double y0 = chix[low];
                double y1 = chix[low+1];
                double val = y0 + ((y1 - y0) / (x1 - x0)) * (red_gnu-x0);
                interp[nu_idx] = val;
            }
        }
        ''', 'interp')


    def binSort(self, target):
        lo = 0
        hi = len(self.nu)-1
        while(lo < hi):
            mid = int(lo + (hi-lo) / 2)
            print(lo, hi, mid)
            if(self.nu[mid] > target):
                hi = mid - 1
            elif(self.nu[mid] < target):
                lo = mid + 1
        print(self.nu[lo], target, self.nu[lo-1])
        return lo
        
    def redshift(self,initial_pos,final_pos,ray_ind,star_vel):
        ind_0 = ray_ind[:,0]
        ind_1 = ray_ind[:,1]
        ind_2 = ray_ind[:,2]
        v_op = star_vel[ind_2]-self.vel[ind_1]
        OP =  initial_pos[ind_2]-final_pos[ind_0]
        v_op_norm = np.linalg.norm(v_op,axis=1)[:,np.newaxis]
        OP_norm = np.linalg.norm(OP,axis=1)[:,np.newaxis]
        v_dot = np.sum(v_op*(OP/OP_norm),axis=1)
        v_proj = (OP/OP_norm)*v_dot[:,np.newaxis]
        red = (c_cgs - np.sign(v_dot)*np.linalg.norm(v_proj,axis=1))/c_cgs
        return red
    
    def load_dust(self):
        d = SphericalDust('hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5')
        nu_d = d.optical_properties.nu
        kdust = np.interp(self.nu,nu_d,d.optical_properties.chi).T
        alb_dust = np.interp(self.nu,nu_d,d.optical_properties.albedo).T
        chisdust = alb_dust*kdust
        chivdust = kdust - chisdust
        self.emiss_dust_0 = np.array([np.interp(self.nu, d.emissivities.nu, np.array(d.emissivities.jnu)[:,i]) \
                                   for i in range(len(d.emissivities.jnu[0]))])
        self.dust_nrg = np.array(d.emissivities.var)
        self.chisdust_0 = chisdust[np.newaxis,:]*(self.temp<self.Tmax)[:,np.newaxis]
        self.chivdust_0 = chivdust[np.newaxis,:]*(self.temp<self.Tmax)[:,np.newaxis]
        #self.P1, self.P2, self.P3, self.P4 = self.get_scattering(d,nu_d)
        
    def load_mock_dust(self):
        chivdust, chisdust = np.zeros_like(self.nu), np.zeros_like(self.nu)
        self.chisdust_0 = chisdust[np.newaxis, :]
        self.chivdust_0 = chivdust[np.newaxis, :]
        

    def ray_trace_1(self, initial_pos, final_pos, cell_based=True):
        ll = self.ll
        ur = self.ur
        split_ll = np.array_split(np.arange(len(self.ll)),np.maximum(len(ll)/200,1))
        M = final_pos[:,np.newaxis]-initial_pos
        ray_ind = np.array([])
        tmin_f = np.array([])
        tmax_f = np.array([])
        i = 0
        for split_ll_i in split_ll:
            t0 = (ll[split_ll_i,np.newaxis]-initial_pos)/M[:,np.newaxis]
            t1 = (ur[split_ll_i,np.newaxis]-initial_pos)/M[:,np.newaxis]
            tmin = np.minimum(t0, t1)
            tmax = np.maximum(t0, t1)
            t0,t1 = 0,0
            tmin = tmin.max(axis=3)
            tmax = tmax.min(axis=3)
            index = np.arange(tmin.shape[1])
            bool_tmin = (tmin < tmax)*(tmin <1)*(tmax>0)
            tmin = tmin[bool_tmin]
            tmax = tmax[bool_tmin]
            target_ind,cell_ind,star_ind = np.where(bool_tmin)
            bool_tmin = 0
            if len(ray_ind) == 0:
                ray_ind = np.stack((target_ind,split_ll_i[cell_ind],star_ind),axis=1)
            else:
                ray_ind = np.vstack((ray_ind,np.stack((target_ind,split_ll_i[cell_ind],star_ind),axis=1)))
            target_ind,cell_ind,star_ind =0,0,0
            tmin_f = np.append(tmin_f,tmin)
            tmax_f = np.append(tmax_f,tmax)
            tmin,tmax = 0,0
            i += 1
        tmin_f = np.maximum(tmin_f,0)
        if not cell_based:
            tmax_f = np.minimum(tmax_f,1)
        p_close = tmin_f[:,np.newaxis]*M[ray_ind[:,0],ray_ind[:,2]]+initial_pos[ray_ind[:,2]]
        p_far = tmax_f[:,np.newaxis]*M[ray_ind[:,0],ray_ind[:,2]]+initial_pos[ray_ind[:,2]]
        dr = np.linalg.norm(p_far-p_close, axis=1)
        return dr,ray_ind
    
    def par_ray_trace_1(self, initial_pos, final_pos):
        ll = self.ll
        ur = self.ur
        ll_g = cp.asarray(ll, dtype=cp.float64)
        ur_g = cp.asarray(ur, dtype=cp.float64)
        ipos_g = cp.asarray(initial_pos, dtype=cp.float64)
        fpos_g = cp.asarray(final_pos, dtype=cp.float64)
        M = (cp.expand_dims(fpos_g, axis=1) - ipos_g)
        ll_ind = cp.arange(ll_g.shape[0])
        ll_max = max(ll_g.shape[0]//200, 1)
        split_ll_g = cp.array_split(ary=ll_ind, indices_or_sections=ll_max)
        ray_ind = cp.array([], dtype=cp.int16)
        ray_ind_list = []
        tmin_list = []
        tmax_list = []
        for split_ll_i in split_ll_g:
            t0 = (cp.expand_dims(ll_g[split_ll_i], axis=1) - ipos_g) / cp.expand_dims(M, axis=1) 
            t1 = (cp.expand_dims(ur_g[split_ll_i], axis=1) - ipos_g) / cp.expand_dims(M, axis=1) 
            tmin = cp.minimum(t0, t1) 
            tmax = cp.maximum(t0, t1)
            del t0, t1
            tmin = cp.max(tmin, axis=3)
            tmax = cp.min(tmax, axis=3)
            bool_tmin = self.bool_tmin_kernel(tmax, tmin)
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
        del tmin_f
        ray_ind_col2 = ray_ind[:, 2]
        ray_ind_col0 = ray_ind[:,0]
        p_close = cp.expand_dims(tmin_f_clamped, axis=1)*M[ray_ind_col0,ray_ind_col2]+ipos_g[ray_ind_col2]
        p_far = cp.expand_dims(tmax_f, axis=1)*M[ray_ind_col0,ray_ind_col2]+ipos_g[ray_ind_col2]
        dr = cp.linalg.norm(p_far-p_close, axis=1)
        return dr.get(), ray_ind.get()
    
    
    def exp_prt_4(self, ray_ind, dr, ipos, fpos):
        gtau_i_j = cp.zeros((len(fpos), len(ipos), len(self.nu)), dtype=cp.float64)
        gredshift = cp.asarray(self.redshift(ipos, fpos, ray_ind, self.svels), dtype=cp.float64)
        gnu = cp.asarray(self.nu, dtype=cp.float64)
        gmetals = cp.asarray(self.metals, dtype=cp.float64)
        gi_temp = cp.asarray(self.i_temp)
        gray_ind = cp.asarray(ray_ind)
        # gray_ind_col_1 = cp.asarray(ray_ind[:,0])
        # gray_ind_col_2 = cp.asarray(ray_ind[:,1])
        # gray_ind_col_3 = cp.asarray(ray_ind[:,2])
        gdr = cp.asarray(dr, dtype=cp.float64)
        gchishe = cp.asarray(self.chishe, dtype=cp.float64)
        gchismet = cp.asarray(self.chismet_0, dtype=cp.float64)
        gchivhe = cp.asarray(self.chivhe, dtype=cp.float64)
        gchivmet = cp.asarray(self.chivmet_0, dtype=cp.float64) 
        gden = cp.asarray(self.den, dtype=cp.float64)
        #define kernel constants
        n_nu = len(gnu)
        
        n_fpos = gtau_i_j.shape[1]
        threads_per_block = 256
        diff_nu = cp.min(cp.abs(cp.diff(gnu)/gnu[1:]))
        bool_red = self.bool_red_kernel(gredshift, diff_nu)
        ind_true = cp.arange(len(gmetals))
        ind_all = cp.unique(gray_ind[:,1][cp.isin(gray_ind[:, 1],ind_true)])
        ray_ind_arange = cp.arange(len(ray_ind))
        temp_ind = cp.unique(gi_temp[ind_all])
        print(len(ind_all))
        split_inds = cp.array_split(ind_all, int(max(len(ind_all) / 200, 1)))
        bool_in_sum = cp.zeros(gray_ind.shape[0])
        #print(bool_in_sum.shape)
        bigcount = 0
        for ind_i, inds in enumerate(split_inds):
            count = 0
            temp_j = cp.minimum(cp.searchsorted(temp_ind, gi_temp[inds]), len(temp_ind)-1)
            Z = cp.expand_dims(gmetals[inds], axis=1)
            chiden = (cp.expand_dims(gden[inds], axis=1) / mH)
            # chix = self.chix_kernel(Z, gchishe[temp_j], gchismet[temp_j], gchivhe[temp_j], gchivmet[temp_j], chiden)
            chix = gchivhe[temp_j] * chiden
            bool_in = cp.isin(gray_ind[:, 1], inds)
            ray_ind_i = ray_ind_arange[bool_in]
            i_s, t_s, j_s = gray_ind[bool_in][:,0],gray_ind[bool_in][:,1],gray_ind[bool_in][:,2]
            print(len(i_s), len(j_s), len(t_s))
            n_rays = len(i_s)
            chi_ind = cp.searchsorted(inds, t_s)
            drt = gdr[bool_in]
            # len_y = cp.arange(len(i_s))
            # bool_y_red = bool_red[ray_ind_i]
            # cp.logical_not(bool_y_red, out=bool_y_red)
            blocks_per_ray = (n_nu + threads_per_block - 1) // threads_per_block
            grid = (n_rays, blocks_per_ray)
            block = (threads_per_block,)
            red = cp.ones_like(gredshift[ray_ind_i])
            #gredshift[cp.abs(gredshift - 1.0) < 1e-14] = 1.0
            #batch_interp_kernel expects int32
            chi_ind32 = chi_ind.astype(cp.int32)
            i_s32 = i_s.astype(cp.int32)
            j_s32 = j_s.astype(cp.int32)

            self.batch_interp_kernel(
                grid, block,
                (gnu, red, chix, chi_ind32, drt, i_s32, j_s32, 
                n_nu, n_rays, n_fpos, gtau_i_j)
            )
            # cp.cuda.Stream.null.synchronize()
            # # cupyx.scatter_add(gtau_i_j, (i_s[bool_y_red], j_s[bool_y_red]), cp.expand_dims(drt[bool_y_red], axis=1)*chix[chi_ind[bool_y_red]])
            # # for y in len_y[bool_red[ray_ind_i]]:
            # #     chix_t = cp.interp(gnu,gnu*gredshift[ray_ind_i[y]],chix[chi_ind[y]])
            # #     gtau_i_j[i_s[y],j_s[y]] += drt[y]*chix_t
            bool_in_sum += bool_in
            # del chix

        bigcount = bool_in_sum.sum()
        cp.exp(-gtau_i_j, out=gtau_i_j)
        return gtau_i_j.get(), bigcount
    
    def ray_trace_4_cpu(self,ray_ind,dr,final_pos,initial_pos):
        tau_i_j = np.zeros((len(final_pos),len(initial_pos),len(self.nu)))
        # red = self.redshift(initial_pos,final_pos,ray_ind,self.svels)
        red = np.ones(len(ray_ind))
        #ind_all = np.unique(ray_ind[:,1])
        bigcount = 0
        #time_piece = np.zeros(2)
        diff_nu = np.abs(np.diff(self.nu)/self.nu[1:]).min()
        bool_red = np.abs(red-1) > 0.5*diff_nu
        np.arange(len(self.metals))
        ind_true = np.arange(len(self.metals))
        ind_all_0 = ray_ind[:,1][np.isin(ray_ind[:,1],ind_true)]
        if len(ind_all_0 )>0:
            ind_all = np.unique(ind_all_0)
            self.temp_ind = np.unique(self.i_temp[ind_all])
            split_inds = np.array_split(ind_all,max(len(ind_all)/200,1))
            i_s, j_s = None, None
            for ind_i,inds in enumerate(split_inds):
                count = 0
                chix = {}
                temp_j = np.minimum(np.searchsorted(self.temp_ind,self.i_temp[inds]),len(self.temp_ind)-1)
                # timei = time.time()
                Z = self.metals[inds][:,np.newaxis]
                # DGRm = mH*10**(2.445*np.log10(Z)-2.029)
                # chix = self.chishe[temp_j] + 0.0204*Z*self.chismet_0[temp_j] +\
                #                                 self.chivhe[temp_j] + 0.0204*Z*self.chivmet_0[temp_j]
                chix = self.chivhe[temp_j]
                chix *= self.den[inds,np.newaxis]/mH
                #print(chix[0,0])
                bool_in = np.isin(ray_ind[:,1],inds)
                ray_ind_i = np.arange(len(ray_ind))[bool_in]
                i_s, t_s, j_s = ray_ind[bool_in][:,0],ray_ind[bool_in][:,1],ray_ind[bool_in][:,2]
                chi_ind = np.searchsorted(inds,t_s)
                drt = dr[bool_in]
                len_y = np.arange(len(i_s))
                bool_y_red = np.logical_not(bool_red[ray_ind_i])
                np.add.at(tau_i_j,(i_s[bool_y_red],j_s[bool_y_red]),drt[bool_y_red,np.newaxis]*(chix[chi_ind[bool_y_red]]))
                for y in len_y[bool_red[ray_ind_i]]:
                        chix_t = np.interp(self.nu,self.nu*red[ray_ind_i[y]],chix[chi_ind[y]])
                        tau_i_j[i_s[y],j_s[y]] += drt[y]*chix_t
                count += bool_in.sum()
                bigcount += count
                chix = None
        #print(tau_i_j[0,0,5])
        tau_i_j = np.exp(-tau_i_j)

        return tau_i_j,bigcount,i_s, j_s


if __name__ == "__main__":
    c = 2.998e+18
    mH = 1.67e-24
    pc = 3.086e+18
    sigma = 5.67051e-5
    c_cgs =  2.99792458e10
    MIN_DENSITY = 1e-27
    PROTON_MASS = 1.67262192e-24
    halo_version = 2020 #updated from local file
    h_0 = Halo(0)

    job_split = np.array_split(np.arange(len(h_0.spos)),np.maximum(len(h_0.spos)/4,1))
    i_stars = job_split[0]
    print(len(i_stars))
    ipos_test = h_0.spos[i_stars]
    fpos_test = ((h_0.ll+h_0.ur)/2)[i_stars]
    print(ipos_test.shape)

    # print('Running CPU-Side Test: Ray Trace 1')
    # s_t  = time.time()
    # dr_cpu, ray_ind_cpu = h_0.ray_trace_1(initial_pos=ipos_test, final_pos=fpos_test)
    # print('CPU run complete. Took {}s'.format(time.time() - s_t))

    # chix = cp.array(h_0.chivhe[30], dtype=cp.float64)
    # nu = cp.array(h_0.nu, dtype=cp.float64)
    # red = cp.float64(2)
    # n_nu = len(nu)
    # out_arr = cp.zeros_like(h_0.nu)
    # threads_per_block = 256
    # blocks_per_ray = (n_nu +threads_per_block - 1) // threads_per_block
    # grid = (blocks_per_ray, 1)
    # block = (threads_per_block, )
    # plt.figure()
    # plt.xlabel('Wavelength')
    # plt.ylabel('Redshifted chivhe (GPU)')
    # plt.semilogx()
    # for i in range(1, 11):
    #     red = cp.float64(i)
    #     h_0.interp_kernel(grid, block, (nu, red, chix, n_nu, out_arr))
    #     out_cpu = out_arr.get()
    #     out_numpy = np.interp(h_0.nu, h_0.nu*i, h_0.chivhe[30])
    #     plt.plot(h_0.nu[h_0.nu > 1e14], np.abs(out_cpu[h_0.nu > 1e14] - out_numpy[h_0.nu > 1e14]) / out_numpy[h_0.nu > 1e14],  label = str('Redshift = ' + str(i)))
    # plt.legend()
    # plt.show()



    if (cp.cuda.is_available()):
        print("GPU available: {}".format(cp.cuda.Device().id))
    print('Running GPU-Side Test: Ray Trace 1')
    s_t  = time.time()
    dr_gpu, ray_ind_gpu = h_0.par_ray_trace_1(initial_pos=ipos_test, final_pos=fpos_test)
    print('GPU run complete. Took {}s'.format(time.time() - s_t))

    # # print(np.abs(dr_cpu - dr_gpu))
    # # print(np.abs(ray_ind_cpu - ray_ind_gpu))

    print('Running GPU-Side Test: Ray Trace 4')
    s_t  = time.time()
    tau_gpu, bigcount = h_0.exp_prt_4(ray_ind_gpu, dr_gpu, ipos_test, fpos_test)
    print('GPU run complete. Took {}s'.format(time.time() - s_t))

    print('Running CPU-Side Test: Ray Trace 4')
    s_t = time.time()
    tau_cpu, bigcount_cpu, i_s, j_s = h_0.ray_trace_4_cpu(ray_ind_gpu, dr_gpu, ipos_test, fpos_test)
    print('CPU run complete. Took {}s'.format(time.time() - s_t))
    print(bigcount, bigcount_cpu)
    
    print(tau_cpu.shape, tau_gpu.shape)
    plt.figure()
    fig, ax = plt.subplots(tau_cpu.shape[0], tau_cpu.shape[1], figsize=(10, 10))
    for axis in ax.flat:
        axis.set(xlabel="Wavelength", ylabel="Absorption")
        axis.semilogx()
    
    for i in range(ipos_test.shape[0]):
        for j in range(fpos_test.shape[0]):
            ax[i, j].plot(h_0.nu, tau_cpu[i][j], alpha=0.4)
            ax[i, j].plot(h_0.nu, tau_gpu[i][j], alpha=0.4)
            ax[i, j].set_title('Tau [{}, {}]'.format(i, j))
    
    # plt.plot(h_0.nu, np.abs(tau_cpu[1][0]), alpha=0.4, label="CPU")
    # plt.plot(h_0.nu, np.abs(tau_gpu[1][0]), alpha=0.4, label="GPU")
    # fig.suptitle('{}x{} Tau grid: CPU-GPU comparison'.format(tau_cpu.shape[0], tau_cpu.shape[1]))
    plt.legend()
    # fig.tight_layout() 
    plt.show()
    

    # print(benchmark(h_0.par_ray_trace_4, args=(ray_ind_gpu, dr_gpu, ipos_test, fpos_test), n_repeat=10))
    # print(benchmark(h_0.ray_trace_4, args=(ray_ind_gpu, dr_gpu, ipos_test, fpos_test), n_repeat=10))

