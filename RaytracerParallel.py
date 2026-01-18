import yt
import numpy as np
import pandas as pd
import time
import cupy as cp
from cupy import fuse
import cupyx
from cupyx.profiler import benchmark, profile
from hyperion.dust import SphericalDust

class Halo:
    def __init__(self):
        halo = np.load('box_3_z_1_halotree.npy',allow_pickle=True).tolist()
        ds = yt.load('box_3_z_1-final/DD0679/output_0679')
        plotd = np.load('plothype0_52.npy',allow_pickle=True).tolist()
        rad = halo['0'][5]['Halo_Radius']
        center = halo['0'][5]['Halo_Center']
        yt_center = ds.length_unit.in_units("cm") * center
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
        self.stars = np.load('starlists_1408.npy',allow_pickle=True).tolist()
        self.star_pos = self.stars['0'][0]['positions2'] * ds.length_unit.in_units("cm").v
        self.svels = self.stars['0'][0]['vels2']
        self.absp = plotd['chivhe']
        self.chismet_0 = plotd['chismet']
        self.chishe = plotd['chishe']
        self.nu = plotd['nu']
        self.temps = reg['temperature'].in_units('K')
        self.i_temp = np.minimum(np.searchsorted(plotd['temp'],self.temps,side='right'),len(plotd['temp'])-1)
        self.absp = plotd['chivhe']
        self.metals = reg['metallicity'].in_units('Zsun').V
        self.ray_end = np.array([(star +( (np.random.rand(1, ) -0.5) *radius.v)) for star in self.star_pos])
        self.ll_box = center-radius.v
        self.ur_box = center+radius.v
        self.den = np.array((reg['HI_Density']+reg['H2I_Density']+reg['H2II_Density']).in_units('g/cm**3').v)
        

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
    
    def ray_trace_1(self, cell_based=True):
        initial_pos = self.star_pos
        final_pos = self.ray_end
        ll = self.ll
        ur = self.ur

        M = final_pos[:,np.newaxis]-initial_pos
        split_ll = np.array_split(np.arange(len(self.ll)),np.maximum(len(ll)/200,1))
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
        print(dr.shape)
        return dr,ray_ind
    
    def par_ray_trace_1(self):
        initial_pos = self.star_pos
        final_pos = self.ray_end
        ll = self.ll
        ur = self.ur

        ll_g = cp.asarray(ll, dtype=cp.float64)
        ur_g = cp.asarray(ur, dtype=cp.float64)
        ipos_g = cp.asarray(initial_pos, dtype=cp.float64)
        fpos_g = cp.asarray(final_pos, dtype=cp.float64)
        M = (cp.expand_dims(fpos_g, axis=1) - ipos_g)
        ll_ind = cp.arange(ll_g.shape[0])
        ll_max = np.maximum(ll_g.shape[0]//200, 1)
        split_ll_g = cp.array_split(ary=ll_ind, indices_or_sections=ll_max)
        tmin_f = cp.array([], dtype=cp.float64)
        tmax_f = cp.array([], dtype=cp.float64)
        ray_ind = cp.array([], dtype=cp.int16)
        i = 0
        for split_ll_i in split_ll_g:
            t0 = (cp.expand_dims(ll_g[split_ll_i], axis=1) - ipos_g) / cp.expand_dims(M, axis=1) 
            t1 = (cp.expand_dims(ur_g[split_ll_i], axis=1) - ipos_g) / cp.expand_dims(M, axis=1) 
            tmin = cp.minimum(t0, t1) 
            tmax = cp.maximum(t0, t1)
            del t0, t1
            tmin = cp.max(tmin, axis=3)
            tmax = cp.min(tmax, axis=3)
            bool_tmin = (tmin < tmax)*(tmin < 1)*(tmax > 0) 
            tmax_b = tmax[bool_tmin]
            del tmax
            tmin_b = tmin[bool_tmin]
            del tmin
            target_ind,cell_ind,star_ind = cp.where(bool_tmin)
            del bool_tmin
            ray_ind_g = cp.stack((target_ind, split_ll_i[cell_ind], star_ind), axis=1)
            del target_ind, cell_ind, star_ind
            tmin_f = cp.append(tmin_f, tmin_b)
            tmax_f = cp.append(tmax_f, tmax_b)
            del tmin_b, tmax_b
            if(i == 0):
                ray_ind = ray_ind_g
            else:
                ray_ind = cp.vstack((ray_ind, ray_ind_g))
            del ray_ind_g
            i += 1
        tmin_f_clamped = cp.maximum(tmin_f, 0)
        del tmin_f
        p_close = cp.expand_dims(tmin_f_clamped, axis=1)*M[ray_ind[:,0],ray_ind[:,2]]+ipos_g[ray_ind[:,2]]
        p_far = cp.expand_dims(tmax_f, axis=1)*M[ray_ind[:,0],ray_ind[:,2]]+ipos_g[ray_ind[:,2]]
        dr = cp.linalg.norm(p_far-p_close, axis=1)
        return dr.get(), ray_ind.get()

    def ray_trace_4(self, ray_ind, dr):
        final_pos = self.ray_end
        initial_pos = self.star_pos
        tau_i_j = np.zeros((len(final_pos),len(initial_pos),len(self.nu)))
        red = self.redshift(initial_pos,final_pos,ray_ind,self.svels)
        ind_all = np.unique(ray_ind[:,1])
        split_inds = np.array_split(ind_all,max(len(ind_all)/100,1))
        bigcount = 0
        diff_nu = np.abs(np.diff(self.nu)/self.nu[1:]).min()
        bool_red = np.abs(red-1) > 0.5*diff_nu
        for ind_i,inds in enumerate(split_inds):
            count = 0
            chix = {}
            temp_j = self.i_temp[inds]
            inds_list = np.arange(len(inds))
            Z = self.metals[inds][:,np.newaxis]
            DGRm = mH*10**(2.445*np.log10(Z)-2.029)
            #need to implement hyperion for chisdust_0 and chivdust_0
            chix = self.chisdust_0[temp_j]*DGRm + self.chishe[temp_j] + Z*self.chismet_0[temp_j] +\
                                        self.chivdust_0[temp_j]*DGRm + self.chivhe[temp_j] + Z*self.chivmet_0[temp_j]
            chix *= self.den[inds,np.newaxis]/mH
            chix = {t: chix[i] for i,t in enumerate(inds)}
            bool_in = np.isin(ray_ind[:,1],inds)
            ray_ind_i = np.arange(len(ray_ind))[bool_in]
            i_s, t_s, j_s = ray_ind[bool_in][:,0],ray_ind[bool_in][:,1],ray_ind[bool_in][:,2]
            drt = dr[bool_in]
            for y in range(len(i_s)):
                if bool_red[ray_ind_i[y]]:
                    chix_t = cp.interp(self.nu,self.nu*red[ray_ind_i[y]],chix[t_s[y]])
                    tau_i_j[i_s[y]][j_s[y]] += drt[y]*chix_t
                else:
                    tau_i_j[i_s[y]][j_s[y]] += drt[y]*chix[t_s[y]]
            count += bool_in.sum()
            bigcount += count
        chix = {}
        tau_i_j = np.exp(-tau_i_j)
        return tau_i_j,bigcount


if __name__ == "__main__":
    c = 2.998e+18
    mH = 1.67e-24
    pc = 3.086e+18
    sigma = 5.67051e-5
    c_cgs =  2.99792458e10
    MIN_DENSITY = 1e-27
    PROTON_MASS = 1.67262192e-24

