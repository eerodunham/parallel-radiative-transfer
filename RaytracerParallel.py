import yt
import numpy as np
import pandas as pd
import time
import cupy as cp

halo = np.load('box_3_z_1_halotree.npy',allow_pickle=True).tolist()
ds = yt.load('box_3_z_1-final/DD0679/output_0679')
rad = halo['0'][5]['Halo_Radius']
center = halo['0'][5]['Halo_Center']
MIN_DENSITY = 1e-27
PROTON_MASS = 1.67262192e-24


# construct a region as an AABB of halo
reg = ds.box((center-rad),(center+rad))
yt_center = ds.length_unit.in_units("cm") * center
radius = ds.length_unit.in_units('cm') * rad
# list of all star positions for raycasting
stars = np.load('starlists_1408.npy',allow_pickle=True).tolist()
star_pos = stars['0'][0]['positions2'] * ds.length_unit.in_units("cm").v
# yt allows us to query the region and obtain all of the highest-level-possible data across the region. Here, we query the xyz positions and 
# concatenate them into a single buffer of lower-left positions for each grid cell.
ll_x = ((reg['x']-reg['dx']/2)).in_units('cm').v
ll_y = ((reg['y']-reg['dy']/2)).in_units('cm').v
ll_z = ((reg['z']-reg['dz']/2)).in_units('cm').v
# we also wil need the ds length of the given cell
ll = np.column_stack((np.column_stack((ll_x,ll_y)),ll_z))
dds = reg['dx'].in_units('cm').v
# finally, we need the density and absorption of the gas at each cell.
plotd = np.load('plothype0_52.npy',allow_pickle=True).tolist()
temp = reg['Temperature'].in_units('K').v
absp = plotd['chivhe']
temp_index = np.minimum(np.searchsorted(plotd['temp'],temp,side='right'),len(plotd['temp'])-1)
# den = np.array(reg['Density'].in_units('g/cm**3'))
den = np.array((reg['HI_Density']+reg['H2I_Density']+reg['H2II_Density']).in_units('g/cm**3').v)
ray_end = np.array([(star +( (np.random.rand(1, ) -0.5) *radius.v)) for star in star_pos])
cam = (np.random.rand(3, ) - 0.5) * 3.086e21 
ll_box = center-radius.v
ur_box = center+radius.v

#cpu ray-trace-1: takes in lower-left & upper-right corners of all cells, and start/end positions of all rays
#computes 
def ray_trace_1(ll,ur,initial_pos,final_pos,cell_based=True):
    print(final_pos.shape, initial_pos.shape)
    M = final_pos[:,np.newaxis]-initial_pos
    split_ll = np.array_split(np.arange(len(ll)),np.maximum(len(ll)/200,1))
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
        print('Iter (cpu): {}'.format(i))
        i += 1
    tmin_f = np.maximum(tmin_f,0)
    if not cell_based:
        tmax_f = np.minimum(tmax_f,1)
    p_close = tmin_f[:,np.newaxis]*M[ray_ind[:,0],ray_ind[:,2]]+initial_pos[ray_ind[:,2]]
    p_far = tmax_f[:,np.newaxis]*M[ray_ind[:,0],ray_ind[:,2]]+initial_pos[ray_ind[:,2]]
    dr = np.linalg.norm(p_far-p_close, axis=1)
    print(dr.shape)
    return dr,ray_ind


def ray_trace_1_parallel(ll, ur, initial_pos, final_pos):
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
        print('Iter: {}'.format(i))
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
    print(dr.shape)
    return dr.get(), ray_ind.get()







print('Lower left shape: {}\nDDS shape: {}\nInit pos shape: {}\nFinal pos shape: {}'.format(ll.shape, dds.shape, star_pos.shape, ray_end.shape))
ur = ll+dds[:, np.newaxis]
# print(ll.shape, dds.shape, ur.shape)
s_tc = time.time()
dr_c, ray_ind_c = ray_trace_1(ll, ur, star_pos, ray_end[:10])
e_tc = time.time()
dr, ray_ind = ray_trace_1_parallel(ll, ur, star_pos, ray_end[:10])
e_tg = time.time()
print('CPU: {}\nGPU: {}'.format(e_tc-s_tc, e_tg-e_tc))
