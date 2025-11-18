import yt
import numpy as np
from SPS_reader import SSP_interpolator
import matplotlib.pyplot as plt
from scipy import special
from scipy import integrate
from itertools import permutations
import time
import sys,os
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy.interpolate import interp1d

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


class Radiative_Transfer():
    def __init__(self,halo,timestep):
        self.halo = halo
        self.timestep = timestep
        if delta:
            self.path_to_fsps = '/work/hdd/bezm/gtg115x/Analysis/fsps/'#'/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/fsps/'
        else:
            self.path_to_fsps = '/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/fsps/'
        self.plot_path = savestring+'/Results_%s_%s/test_plots_%s/' % (halo,timestep,test_num)
        if rank==0:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
        #ds_path = '/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/box_3_z_1/DD0683/output_0683'
        self.Tmax = 1100
        self.star_folder = ds_path_0+savestring+'/'
        ds_path_1 = self.star_folder+'/'+'pfs_allsnaps_%s.txt' % halo_version
        file_list = np.loadtxt(ds_path_1,dtype=str)[:,0]
        self.ds = yt.load(file_list[timestep])
        self.stars = np.load(self.star_folder+'starlists_2013.npy',allow_pickle=True).tolist()[halo][timestep]
        self.get_spos()
        if rank==0:
            self.get_stars()
        #self.lums = None
        self.Spectra = np.array([])
        np.random.seed(seed=10)
        self.randQ_star = np.random.normal(loc=0.01, scale=0.01, size=len(self.spos))
        self.randU_star = np.random.normal(loc=0.01, scale=0.01, size=len(self.spos))
        densities,dx,self.temps,metals,ll,ur,self.vel = self.get_grid_values()
        ind_temps = np.arange(len(self.temps))[self.temps>1e5]
        self.get_gas_rads()
        self.i_temp = np.minimum(np.searchsorted(self.plot['temp'],self.temps),len(self.temp)-1)
        self.load_dust()
        z_now = self.ds.current_redshift
        self.cmb = 0*4*np.pi*self.blackbod(self.nu,np.array((1+z_now)*2.73))/c_cgs
        self.i_cen_range = None
        self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t = None,None,None,None
        if rank ==0:#np.arange(len(dx))#
            if not os.path.exists(self.plot_path+'nu.txt'):
                np.savetxt(self.plot_path+'nu.txt',self.nu)
            if not os.path.exists(self.plot_path+'cells.npy'):
                self.i_cen_range = np.random.choice(np.arange(len(dx)), 100, replace=False)#np.array([1898,1899,1900])
                np.save(self.plot_path+'cells.npy',self.i_cen_range)
            else:
                self.i_cen_range = np.load(self.plot_path+'cells.npy',allow_pickle=True)
            if not os.path.exists(self.plot_path+'properties.npy'):
                properties = {}
                properties['density'] = densities
                properties['temperature'] = self.temps
                properties['metallicity'] = metals
                properties['dx'] = dx
                properties['center'] = (ll+ur)/2
                properties['spos'] = self.spos
                properties['lums'] = self.lums
                np.save(self.plot_path+'properties.npy',properties)
                properties = None
            self.emission = np.zeros((len(self.i_cen_range),4,len(self.nu)))
            self.emiss_pos = np.zeros((len(self.i_cen_range),3))
            self.emiss_vel = np.zeros((len(self.i_cen_range),3))
        self.i_cen_range = comm.bcast(self.i_cen_range, root=0)
        self.randQ = np.random.normal(loc=0.0, scale=0.1,size=len(dx))
        self.randU = np.random.normal(loc=0.0, scale=0.1,size=len(dx))
        self.plot_ind = 0
        self.run_rounds = 2
        for run_round in range(self.run_rounds):
            self.plot_ind = run_round
            self.cell_split = np.array_split(self.i_cen_range,np.maximum(len(self.i_cen_range)/50,1))
            self.new = False
            if rank ==0:
                self.emission_1 = np.zeros((len(self.i_cen_range),4,len(self.nu)))
                self.emiss_pos_1 = np.zeros((len(self.i_cen_range),3))
                self.emiss_vel_1 = np.zeros((len(self.i_cen_range),3))
            if not os.path.exists(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,len(self.cell_split)-1)):
                if rank==0:
                    print('Running round %s of %s' % (run_round+1,self.run_rounds))
                self.run_transfer(ll,ur,metals,dx,densities)
                if rank ==0:
                    self.new = True
                    #print(self.emission)
                    #self.emission *= len(dx)/(len(self.emission))
                    print(np.sum(integrate.simpson(self.emission[:,0,:],self.nu))/3e33)
                z_now = comm.bcast(z_now, root=0)
            elif rank==0:
                len_emiss = 0
                for cell_round in range(len(self.cell_split)):
                    indicies = np.arange(len(self.cell_split[cell_round]))+len_emiss
                    len_emiss += len(self.cell_split[cell_round])
                    self.emission[indicies] = np.load(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round) , allow_pickle=True)
                    self.emiss_pos[indicies] = np.load(self.plot_path+'emiss_pos_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                    self.emiss_vel[indicies] = np.load(self.plot_path+'emiss_vel_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
            if not os.path.exists(self.plot_path+'Final_%s.pdf' % self.plot_ind):
                # if rank ==0:
                #     if not self.new:
                #         self.emission *= len(dx)/(len(self.emission))
                self.set_observer(ll,ur,metals,densities,self.halo_c\
                                +np.array([0,1,0])*self.halo_r)
        self.make_image(ll,ur,np.array([0,1,0])*self.halo_r,metals,densities)


    def run_transfer(self,ll,ur,metals,dx,densities):
            len_emiss = 0
            for cell_round in range(len(self.cell_split)):
                indicies = np.arange(len(self.cell_split[cell_round]))+len_emiss
                if not os.path.exists(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round)):
                    i_cells_2 = self.cell_split[cell_round]
                    if rank ==0:
                        print('Running cell batch %s of %s' % (cell_round+1,len(self.cell_split)))
                        print(i_cells_2,self.i_cen_range)
                    if not cuda:
                        self.ray_trace_2(ll,ur,metals,densities,i_cells_2)
                    else:
                        self.ray_trace_2_single(ll,ur,metals,densities,i_cells_2)
                    # if rank==0:
                    #     print(self.I_f_t.sum(axis=1))
                    self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t = comm.bcast((self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t),root=0)
                    ranks = np.arange(nprocs)
                    jobs,sto = job_scheduler_2(np.arange(len(i_cells_2)))
                    job_i = 0
                    rank_now = 0
                    count = 0
                    Done_in = np.full(nprocs,False)
                    if nprocs>0:
                        Done_in[0] = True
                    while Done_in.min()==0:
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                            rank_now = rank
                            req = comm.isend(rank_now,tag=0,dest=0)
                            req.wait()
                        if rank ==0:
                            req = comm.irecv(tag=0,source=MPI.ANY_SOURCE)
                            rank_now = req.wait()
                            if not job_i  <len(sto):
                                Done_in[rank_now] = True
                            #print(job_i,rank_now)
                            req = comm.isend(job_i,tag=1,dest=rank_now)
                            req.wait()
                            req = comm.isend(Done_in,tag=2,dest=rank_now)
                            req.wait()
                            if Done_in.min()>0:
                                for rank_i in jobs:
                                     comm.Send((Done_in),tag=3,dest=rank_i)
                            if not Done_in[rank_now]:
                                jobs[rank_now].append(job_i)
                            job_i += 1
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                            req = comm.irecv(tag=1,source=0)
                            job_i = req.wait()
                            req = comm.irecv(tag=2,source=0)
                            Done_in = req.wait()
                        if rank >= min(nprocs-1,1) and Done_in[rank]:
                            comm.Recv(Done_in,tag=3,source=0)
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                    # for rank_now in ranks:
                    #     if rank == rank_now:
                    #         for cen_index in jobs[rank]:
                                time_0 = time.time()
                                self.cen_index = job_i
                                self.i_cen = i_cells_2[self.cen_index]
                                self.Z = metals[self.i_cen]
                                self.prepare_incident(dx)
                                self.set_absorb()
                                time_2 = time.time()
                                self.redistribution(densities,dx)
                                self.find_Stokes(densities,dx)
                                self.find_Emitted(densities,dx)
                                self.find_Scattered(densities,dx)
                                self.find_Atten(densities,dx)
                                time_f = time.time()
                                sto[self.cen_index]['emission'] = self.Stokes_f+self.Emitted+self.Scattered
                                sto[self.cen_index]['emission_pos'] = (ll[self.i_cen]+ur[self.i_cen])/2
                                sto[self.cen_index]['emission_vel'] = self.vel[self.i_cen]
                                #self.plotting(dx)
                                #print(integrate.simpson((self.Stokes_f+self.Emitted+self.Scattered)[0],self.nu)/1e33,self.temps[self.i_cen])
                                time_p = time.time()
                                #print(self.i_cen,time_2-time_0,time_f-time_2,time_p-time_f)
                    jobs = comm.bcast(jobs,root=0)
                    for rank_now_i in jobs:
                            for i_cen in jobs[rank_now_i]:
                                sto[i_cen] = comm.bcast(sto[i_cen], root=rank_now_i)
                                if rank ==0:
                                    self.emission_1[indicies[i_cen]] = sto[i_cen]['emission']
                                    self.emiss_pos_1[indicies[i_cen]] = sto[i_cen]['emission_pos']
                                    self.emiss_vel_1[indicies[i_cen]] = sto[i_cen]['emission_vel']
                                else:
                                    sto[i_cen] = None
                    if rank ==0:
                        #print(indicies)
                        #print(self.emission_1[:,0].sum(axis=1))
                        #print(self.emission_1[indicies,0].sum(axis=1))
                        np.save(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round),self.emission_1[indicies])
                        np.save(self.plot_path+'emiss_pos_%s_%s.npy' % (self.plot_ind,cell_round),self.emiss_pos_1[indicies])
                        np.save(self.plot_path+'emiss_vel_%s_%s.npy' % (self.plot_ind,cell_round),self.emiss_vel_1[indicies])
                        np.save(self.plot_path+'I_cell_%s_%s.npy' % (self.plot_ind,cell_round),self.I_f_t)
                else:
                    if rank ==0:
                            self.emission_1[indicies] = np.load(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round) , allow_pickle=True)
                            self.emiss_pos_1[indicies] = np.load(self.plot_path+'emiss_pos_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                            self.emiss_vel_1[indicies] = np.load(self.plot_path+'emiss_vel_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                if rank==0:
                  self.emission[indicies] = self.emission_1[indicies]
                  self.emiss_pos[indicies] = self.emiss_pos_1[indicies]
                  self.emiss_vel[indicies] = self.emiss_vel_1[indicies]
                len_emiss += len(self.cell_split[cell_round])



    def set_observer(self,ll,ur,metals,densities,observer_pos,plotnum=0):
        if rank==0:
            print(self.emission[:,0].mean(axis=1))
            wav = c/self.nu
            wav = wav/1e4
            conv = 1
            labs = ['I','|Q|','|U|','|V|']
            colors = ['red','green','orange','purple','blue','brown']
            self.get_stars()
            self.expand_spectra(self.spectra,np.arange(len(self.spectra)))
            dist_mod = 4*np.pi*np.linalg.norm(self.spos-observer_pos,axis=1)**2
            I,Q,U,V = np.sum(self.Spectra/dist_mod[np.newaxis,:,np.newaxis],axis=1)
            plt.plot(wav,conv*I,':',color=colors[0],label=labs[0]+' Initial',linewidth=0.3)
            plt.plot(wav,conv*np.abs(Q),':',color=colors[1],label=labs[1]+' Initial',linewidth=0.3)
            plt.plot(wav,conv*np.abs(U),':',color=colors[2],label=labs[2]+' Initial',linewidth=0.3)
            plt.plot(wav,conv*np.abs(V),':',color=colors[3],label=labs[3]+' Initial',linewidth=0.3)
            print(np.sum(integrate.simpson(self.Spectra[0,:],self.nu))/3e33)
            self.Spectra = np.array([])
        if not os.path.exists(self.plot_path+'I_%s.npy'  % self.plot_ind):
            self.ray_trace_2(ll,ur,metals,densities,observer_pos[:,np.newaxis],cell_based=False)
            if rank==0:
                np.save(self.plot_path+'I_%s.npy' % self.plot_ind,self.I_f_t)
                np.save(self.plot_path+'Q_%s.npy' % self.plot_ind,self.Q_f_t)
                np.save(self.plot_path+'U_%s.npy' % self.plot_ind,self.U_f_t)
                np.save(self.plot_path+'V_%s.npy' % self.plot_ind,self.V_f_t)
        else:
            if rank==0:
                self.I_f_t = np.load(self.plot_path+'I_%s.npy' % self.plot_ind , allow_pickle=True)
                self.Q_f_t = np.load(self.plot_path+'Q_%s.npy' % self.plot_ind , allow_pickle=True)
                self.U_f_t = np.load(self.plot_path+'U_%s.npy' % self.plot_ind , allow_pickle=True)
                self.V_f_t = np.load(self.plot_path+'V_%s.npy' % self.plot_ind , allow_pickle=True)
        if rank ==0:
            I,Q,U,V = self.I_f_t[0],self.Q_f_t[0],self.U_f_t[0],self.V_f_t[0]
            plt.plot(wav,conv*I,color=colors[0],label=labs[0]+' Final')
            plt.plot(wav,conv*np.abs(Q),color=colors[1],label=labs[1]+' Final')
            plt.plot(wav,conv*np.abs(U),color=colors[2],label=labs[2]+' Final')
            plt.plot(wav,conv*np.abs(V),color=colors[3],label=labs[3]+' Final')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Wavelength [micron]')
            plt.ylabel(r'Intensity [erg/s/cm$^2$/Hz]')
            plt.legend(fontsize='x-small')
            plt.ylim(1e-25,max(2e-17,1.2*conv*I.max()))
            plt.xlim(1e-2,1e3)
        #plt.xlim(1e-2,10)
            plt.savefig(self.plot_path+'Final_%s.pdf' % self.plot_ind)
            plt.clf()

    def make_image(self,ll,ur,observer_vec,metals,densities):
        split_emission = None
        if rank ==0:
            split_emission = np.array_split(np.arange(len(self.emission)+len(self.spos)),max((len(self.emission)+len(self.spos))/500,1))
        split_emission = comm.bcast(split_emission,root=0)
        if not os.path.exists(self.plot_path+'V_f_%s.npy' % (len(split_emission)-1)):
            self.ray_trace_2(ll,ur,metals,densities,observer_vec,cell_based=False,parallel=True)
            if rank==0:
                print('Saving %s file(s)' % len(split_emission))
                for i,split_i in enumerate(split_emission):
                    np.save(self.plot_path+'I_f_%s.npy' % i,self.I_f_t[split_i])
                    np.save(self.plot_path+'Q_f_%s.npy' % i,self.Q_f_t[split_i])
                    np.save(self.plot_path+'U_f_%s.npy' % i,self.U_f_t[split_i])
                    np.save(self.plot_path+'V_f_%s.npy' % i,self.V_f_t[split_i])
        else:
            if rank==0:
                self.I_f_t = np.zeros((len(self.emission)+len(self.spos),len(self.nu)))
                self.Q_f_t = np.zeros((len(self.emission)+len(self.spos),len(self.nu)))
                self.U_f_t = np.zeros((len(self.emission)+len(self.spos),len(self.nu)))
                self.V_f_t = np.zeros((len(self.emission)+len(self.spos),len(self.nu)))
                for i,split_i in enumerate(split_emission):
                    self.I_f_t[split_i] = np.load(self.plot_path+'I_f_%s.npy' % i, allow_pickle=True)
                    self.Q_f_t[split_i] = np.load(self.plot_path+'Q_f_%s.npy' % i, allow_pickle=True)
                    self.U_f_t[split_i] = np.load(self.plot_path+'U_f_%s.npy' % i, allow_pickle=True)
                    self.V_f_t[split_i] = np.load(self.plot_path+'V_f_%s.npy' % i, allow_pickle=True)
        if rank ==0:
            wav_bands = np.zeros((3,2))
            wav_bands[0] = [0.62,0.7]
            wav_bands[1] = [0.52,0.56]
            wav_bands[2] = [0.45,0.49]
            Final_source = np.zeros((len(self.I_f_t),3))
            Final_source = convert_RGB_2(self.I_f_t,self.nu,wav_bands)
            #print(Final_source[3])
            num_pix = 500
            wav = c/self.nu
            wav = wav/1e4
            self.halo_r2 = 1*self.spos.std(axis=0).mean()#self.halo_r/30
            #print(self.halo_r2,self.spos.mean(axis=0))
            pix_cen = np.linspace(-self.halo_r2+2*self.halo_r2/num_pix,self.halo_r2-2*self.halo_r2/num_pix,num_pix)
            self.get_stars()
            observer_pos =  np.average(self.spos,axis=0,weights=self.lums)+observer_vec
            xx,yy = np.meshgrid(pix_cen,pix_cen)
            xi,yi = np.meshgrid(np.arange(num_pix),np.arange(num_pix))
            indicies = np.stack((xi,yi),axis=2)
            mesh = np.stack((0*xx,xx,yy),axis=2)
            mesh = self.rot_x(np.array([1,0,0]),observer_vec,mesh)+observer_pos
            o_n = observer_vec/np.linalg.norm(observer_vec)
            d = np.dot(observer_pos,o_n)
            cen = (ll[self.i_cen_range]+ur[self.i_cen_range])/2
            #cen_low = cen + observer_vec
            #cen_high = cen + observer_vec
            ll_in = ll[self.i_cen_range]#(ll[self.i_cen_range]+cen)/2
            ur_in = ur[self.i_cen_range]#(ur[self.i_cen_range]+cen)/2
            permuts = np.unique(np.array(list(permutations([0,1,0,1,0,1],3))),axis=0)
            edges = np.where((permuts)[:, None], ur_in, ll_in)
            tcen = -(np.tensordot(o_n, cen,axes=(0,1)) - d) /np.dot(o_n,observer_vec)
            inter_cen = cen+observer_vec*tcen[:,np.newaxis]
            Pixel = np.zeros((num_pix,num_pix,3))
            for j in range(len(ll_in)):
                hull_v = edges[:,j]
                t = -(np.tensordot(o_n, hull_v,axes=(0,1)) - d) /np.dot(o_n,observer_vec)
                if t.min() >0 and t.max()<2:
                    inter = hull_v+observer_vec*t[:,np.newaxis]
                    hull = ConvexHull(inter,qhull_options='QJ')
                    bool_pix = contained(mesh,hull)
                    if bool_pix.sum()>0:
                        #print(j,bool_pix.sum()/(num_pix**2))
                        #print(Pixel[indicies[bool_pix],:].shape)
                        #idx_0,idx_1 = indicies[bool_pix]
                        #print(Pixel[bool_pix].shape)
                        Pixel[bool_pix] += 1/bool_pix.sum() * Final_source[j+len(self.spos)]
                        # for i in range(bool_pix.sum()):
                        #         idx_0,idx_1 = indicies[bool_pix][i]
                        #         #print(0,idx_0,idx_1)
                        #         Pixel[idx_0][idx_1] += 1/bool_pix.sum() * Final_source[j+len(self.spos)]
                        #print(Pixel[idx_0][idx_1][0].sum())
            tspos = -(np.tensordot(o_n, self.spos,axes=(0,1)) - d) /np.dot(o_n,observer_vec)
            inter2 = self.spos+observer_vec*tspos[:,np.newaxis]
            #print(self.spos)
            for j in range(len(inter2)):
                dist = np.linalg.norm(inter2[j][np.newaxis,np.newaxis,:]-mesh,axis=2)
                min_dist = np.linalg.norm(inter2[j][np.newaxis,np.newaxis,:]-mesh,axis=2).min()
                bool_pix = (dist==min_dist)*(dist< self.halo_r2/num_pix)
                if bool_pix.sum()>0:
                    idx_0,idx_1 = indicies[bool_pix][0]
                    #print(2,idx_0,idx_1)
                    Pixel[idx_0][idx_1] += Final_source[j]
                        #print(Pixel[idx_0][idx_1][0].sum())
                        #print(Pixel[indicies[bool_pix][0],0,:])
            #print(Pixel[:,:,0,:][Pixel[:,:,0,:].sum(axis=2)>0])
            #print(Pixel[:,:,:][Pixel[:,:,:].sum(axis=2)>0])

            Pixel[Pixel>0] = np.log10(Pixel[Pixel>0]) - np.log10(Pixel[Pixel>0]).max()+3
            Pixel[Pixel<0] = 0
            #Pixel = Pixel**(1/2)
            Pixel /= (Pixel.max()-0.2)
            Pixel = np.minimum(Pixel,1)
            #Pixel = Pixel**(1/2)
            #print(Pixel[:,:,:][Pixel[:,:,:].sum(axis=2)>0])
            #Pixel = integrate.simpson(Pixel[:,:,0,:],self.nu)
            from PIL import Image
            plt.imshow(Pixel[:,:,:],origin='lower',interpolation='spline16')
            plt.axis('off')
            plt.savefig(self.plot_path+'image_%s.pdf' % self.plot_ind,bbox_inches='tight',pad_inches=0)
            plt.clf()
            #np.save(self.plot_path+'Pixel_info.npy',Pixel)



    def rot_x(self,a,b,v):
        b = b/np.linalg.norm(b)
        a_b = np.dot(a,b)
        if np.linalg.norm(np.cross(a, b)) >0:
            x,y,z = np.cross(a, b)
        else:
            x,y,z = a
        c = a_b
        s = np.sqrt(1-c*c)
        C = 1-c
        rmat = np.array([[ x*x*C+c,    x*y*C-z*s,  x*z*C+y*s ],\
              [ y*x*C+z*s, y*y*C+c,    y*z*C-x*s ],\
              [ z*x*C-y*s,  z*y*C+x*s,  z*z*C+c   ]])
        rmat[np.abs(rmat)<1e-10] = 0
        indicies = []
        print()
        f_m = np.tensordot(v,rmat,axes=(v.ndim-1,0))
        f_m[np.abs(f_m)<1e-10] = 0
        return f_m

    def prepare_incident(self,dx):
        self.I_f = self.I_f_t[self.cen_index]*(6*dx[self.i_cen]**2)/4
        self.Q_f = self.Q_f_t[self.cen_index]*(6*dx[self.i_cen]**2)/4
        self.U_f = self.U_f_t[self.cen_index]*(6*dx[self.i_cen]**2)/4
        self.V_f = self.V_f_t[self.cen_index]*(6*dx[self.i_cen]**2)/4
        time_1 = time.time()
        np.random.seed(seed=self.i_cen)
        self.cmb_randQ = np.random.normal(loc=0.0, scale=0.1)
        self.cmb_randU = np.random.normal(loc=0.0, scale=0.1)
        self.cmb_randV = np.random.normal(loc=0.0, scale=1e-14)
        mod_cmb = dx[self.i_cen]**3
        self.I_f += self.cmb*mod_cmb
        self.Q_f += self.cmb*self.cmb_randQ*mod_cmb*1e-6/((1+self.ds.current_redshift)*2.73)
        self.U_f += self.cmb*self.cmb_randU*mod_cmb*1e-6/((1+self.ds.current_redshift)*2.73)
        self.V_f += self.cmb*self.cmb_randV*mod_cmb

    def redistribution(self,densities,dx):
        self.redistem = np.zeros(len(self.nu))
        self.redistI = np.zeros(len(self.nu))
        self.redistQ = np.zeros(len(self.nu))
        self.redistU = np.zeros(len(self.nu))
        self.redistV = np.zeros(len(self.nu))
        pe_lim = 0.01
        pe = np.exp(-self.chis[self.i_temp[self.i_cen]]*dx[self.i_cen]*densities[self.i_cen]/mH)
        Ns = np.arange(200)+1
        Pns = pe*(1-pe)**(Ns[:,np.newaxis]-1)
        Ns_all = np.sum(Pns*(Ns[:,np.newaxis]-1),axis=0)
        self.lams = 1/(self.chis[self.i_temp[self.i_cen]]*densities[self.i_cen]/mH)
        self.rp = np.minimum(self.lams,dx[self.i_cen])*Ns_all.astype(int)+dx[self.i_cen]
        self.Ns_all = Ns_all
        self.set_emiss(densities,dx)
        vd = 1.36*np.abs(self.nu[pe< pe_lim]-self.nu[pe>.9][:,np.newaxis])
        if len(vd) >0:
            vd = vd.min(axis=0)
            diff_nu = np.abs(self.nu[:,np.newaxis]-self.nu[pe< pe_lim])/vd
            erf = special.erfc(diff_nu)
            #diff_nu[diff_nu>25] = 25
            R = np.sqrt(np.pi)*erf*np.exp(diff_nu**2)/2
            R[diff_nu>24.9] = 0
            R[pe<0.2] =0
            R = R/R.sum(axis=0)
            self.f_em = np.minimum(1,6*self.lams/dx[self.i_cen])
            self.redistem = (self.emiss[pe< pe_lim][np.newaxis,:]*R).sum(axis=1)
            self.redistI = (self.I_f[pe< pe_lim][np.newaxis,:]*R).sum(axis=1)
            self.redistQ = (self.Q_f[pe< pe_lim][np.newaxis,:]*R).sum(axis=1)
            self.redistU = (self.U_f[pe< pe_lim][np.newaxis,:]*R).sum(axis=1)
            self.redistV = (self.V_f[pe< pe_lim][np.newaxis,:]*R).sum(axis=1)
        pe_dust = np.exp(-self.chisdust[self.i_temp[self.i_cen]]*dx[self.i_cen]*densities[self.i_cen]/mH)
        Pns = pe_dust*(1-pe_dust)**(Ns[:,np.newaxis]-1)
        self.dust_lams = 1/(self.chisdust[self.i_temp[self.i_cen]]*densities[self.i_cen]/mH)
        self.Ns_all_dust = np.sum(Pns*(Ns[:,np.newaxis]-1),axis=0)
        self.Ns_all_dust[pe_dust<1e-3] = 100


    def get_scattering(self,d,nu_d):
        P1 = np.array([np.interp(self.nu, nu_d, d.optical_properties.P1[:,i]) \
                           for i in range(len(d.optical_properties.P1[0]))])
        P2 = np.array([np.interp(self.nu, nu_d, d.optical_properties.P2[:,i]) \
                           for i in range(len(d.optical_properties.P2[0]))])
        P3 = np.array([np.interp(self.nu, nu_d, d.optical_properties.P3[:,i]) \
                           for i in range(len(d.optical_properties.P3[0]))])
        P4 = np.array([np.interp(self.nu, nu_d, d.optical_properties.P4[:,i]) \
                           for i in range(len(d.optical_properties.P4[0]))])
        norm = np.abs(P1).sum(axis=0)
        norm2 = np.abs(P2).sum(axis=0)/norm
        norm3 = np.abs(P3).sum(axis=0)/norm
        norm4 = np.abs(P4).sum(axis=0)/norm
        P1 = P1/norm
        P2 = P2/norm
        P3 = P3/norm
        P4 = P4/norm
        P1_t = P1.sum(axis=0)
        P2_t = P2.mean(axis=0)
        P3_t = P3.mean(axis=0)
        P4_t = P4.mean(axis=0)
        return P1_t, P2_t, P3_t, P4_t

    def load_dust(self):
        from hyperion.dust import SphericalDust
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
        self.P1, self.P2, self.P3, self.P4 = self.get_scattering(d,nu_d)

    def mean_k_intensity(self,kappa):
        mean = integrate.simpson(self.I_f*kappa,self.nu)/integrate.simpson(self.I_f,self.nu)
        return mean

    def ray_trace_2_single(self,ll,ur,metals,densities,i_cells_2,cell_based=True,parallel=False):
        if rank ==0:
            if cell_based:
                final_pos = (ll[i_cells_2]+ur[i_cells_2])/2
                l_final_pos = len(i_cells_2)
                l_final_pos_0 = np.arange(l_final_pos)
                l_final_pos_1 = np.arange(l_final_pos)
                self.offset = 0
                self.cells = i_cells_2
            elif not parallel:
                final_pos = i_cells_2.T
                l_final_pos = 1
                l_final_pos_0 = np.arange(l_final_pos)
                l_final_pos_1 = np.arange(l_final_pos)
                self.offset = 0
            elif parallel:
                l_final_pos_1 = np.arange(len(self.i_cen_range))
                l_final_pos_0 = np.arange(len(self.spos))
                l_final_pos = len(l_final_pos_0) + len(l_final_pos_1)
                self.offset = len(self.spos)
            self.offset_2 = 0
            ranks = np.arange(nprocs)
            self.emiss_ind = []
            self.I_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.Q_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.U_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.V_f_t = np.zeros((l_final_pos,len(self.nu)))
            index = np.arange(len(ll))
            job_split = np.array_split(np.arange(len(self.spos)),np.maximum(len(self.spos)/500,1))
            self.new_stars = True# len(self.Spectra) == 0
            sto = {}
            batch = 1
            if parallel:
                final_pos_0 = self.spos+i_cells_2
            for job_split_i in range(len(job_split)):
                        i_stars = job_split[job_split_i]
                        star_vel = self.svel[i_stars]
                        initial_pos = self.spos[i_stars]
                        if parallel:
                            final_pos = final_pos_0[i_stars]
                            l_pos = len(final_pos)
                        else:
                            l_pos = l_final_pos
                        if self.new_stars:
                            self.get_stars(subset=i_stars)
                        print('Running batch',batch,'of',len(job_split),'on processor',rank)
                        batch += 1
                        #print(len(final_pos),len(initial_pos))
                        sto[job_split_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos,\
                                            star_vel,i_stars,metals,densities,\
                                            np.arange(l_pos),cell_based=cell_based,\
                                            parallel=parallel,offset=0,s_emiss=False)
                        i_range = sto[job_split_i]['i']
                        for i in np.arange(len(i_range)):
                            if sto[job_split_i]['I'][i].sum()>0:
                                #print(i,sto[job_split_i]['I'][i].sum())
                                self.I_f_t[i_range[i]] += sto[job_split_i]['I'][i]
                                self.Q_f_t[i_range[i]] += sto[job_split_i]['Q'][i]
                                self.U_f_t[i_range[i]] += sto[job_split_i]['U'][i]
                                self.V_f_t[i_range[i]] += sto[job_split_i]['V'][i]
                            else:
                                sto[job_split_i] =None
                        sto[job_split_i] = None
                #print(len(self.emission),self.emission.sum())
            if self.plot_ind >0 or not cell_based:
                    final_pos_0 = None
                    job_split = None
                    j_split = None
                    if rank ==0:
                        if parallel:
                            final_pos_0 = self.emiss_pos+i_cells_2
                        else:
                            final_pos_0 = None
                    if rank ==0:
                        j_split = np.array_split(np.arange(len(self.emiss_pos)),np.maximum(len(self.emiss_pos)/1000,nprocs))
                    final_pos_0,j_split = comm.bcast((final_pos_0,j_split),root = 0)
                    for i_jsplit,j_split_i in enumerate(j_split):
                        if rank ==0:
                            print('Running emission group %s of %s' % (i_jsplit+1,len(j_split)))
                        job_split = np.array_split(j_split_i,np.maximum(len(j_split_i)/100,max(1,nprocs)))
                        jobs,sto = job_scheduler(np.arange(len(job_split)),ranklim=len(ranks))
                        batch = 1
                        rank_ids = {}
                        len_ranks = 0
                        sto = {}
                        for job_split_i in range(len(job_split)):
                                rank_ids[job_split_i] = np.array(range(len_ranks,len_ranks+len(job_split[job_split_i])))
                                len_ranks += len(job_split[job_split_i])
                        for job_split_i in range(len(job_split)):
                                    self.emiss_ind = rank_ids[job_split_i]
                                    star_vel = self.emiss_vel[self.emiss_ind]
                                    initial_pos = self.emiss_pos[self.emiss_ind]
                                    if parallel:
                                        final_pos = final_pos_0[i_stars]
                                        l_pos = len(final_pos)
                                    else:
                                        l_pos = l_final_pos
                                    #print(final_pos)
                                    print('Running emission batch',batch,'of',len(job_split),'on processor',rank)
                                    batch += 1
                                    sto[job_split_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos,\
                                                        star_vel,i_stars,metals,densities,\
                                                        np.arange(l_pos),offset=self.offset+self.offset_2,cell_based=cell_based,\
                                                            s_emiss=True,parallel=parallel)
                                    i_range = sto[job_split_i]['i']
                                    for i in np.arange(len(i_range)):
                                        if sto[job_split_i]['I'][i].sum()>0:
                                            #print(i,sto[job_split_i]['I'][i].sum())
                                            self.I_f_t[i_range[i]] += sto[job_split_i]['I'][i]
                                            self.Q_f_t[i_range[i]] += sto[job_split_i]['Q'][i]
                                            self.U_f_t[i_range[i]] += sto[job_split_i]['U'][i]
                                            self.V_f_t[i_range[i]] += sto[job_split_i]['V'][i]
                                        else:
                                            sto[job_split_i] =None
                                    sto[job_split_i] = None

    def ray_trace_2(self,ll,ur,metals,densities,i_cells_2,cell_based=True,parallel=False):
        if cell_based:
            final_pos = (ll[i_cells_2]+ur[i_cells_2])/2
            l_final_pos = len(i_cells_2)
            l_final_pos_0 = np.arange(l_final_pos)
            l_final_pos_1 = np.arange(l_final_pos)
            self.offset = 0
            self.cells = i_cells_2
        elif not parallel:
            final_pos = i_cells_2.T
            l_final_pos = 1
            l_final_pos_0 = np.arange(l_final_pos)
            l_final_pos_1 = np.arange(l_final_pos)
            self.offset = 0
        elif parallel:
            l_final_pos_1 = np.arange(len(self.i_cen_range))
            l_final_pos_0 = np.arange(len(self.spos))
            l_final_pos = len(l_final_pos_0) + len(l_final_pos_1)
            self.offset = len(self.spos)
        self.offset_2 = 0
        ranks = np.arange(nprocs)
        self.emiss_ind = []
        self.I_f_t = np.zeros((l_final_pos,len(self.nu)))
        self.Q_f_t = np.zeros((l_final_pos,len(self.nu)))
        self.U_f_t = np.zeros((l_final_pos,len(self.nu)))
        self.V_f_t = np.zeros((l_final_pos,len(self.nu)))
        index = np.arange(len(ll))
        if parallel:
            final_pos_0 = self.spos+i_cells_2
        j_split = None
        j_split = np.array_split(np.arange(len(self.spos)),np.maximum(len(self.spos)/500,1))
        if rank !=0:
            self.freq = None
        self.freq = comm.bcast(self.freq,root=0)
        for i_jsplit,j_split_i in enumerate(j_split):
            job_split = np.array_split(j_split_i,np.maximum(len(j_split_i)/50,max(1,nprocs-1)))
            jobs,sto = job_scheduler_2(np.arange(len(job_split)))
            self.new_stars = True# len(self.Spectra) == 0
            if rank ==0:
                print('Running stellar group %s of %s (%s Stars)' % (i_jsplit+1,len(j_split),len(j_split_i)))
                time_2 = time.time()
            batch = 1
            job_i = 0
            rank_now = 0
            count = 0
            Done_in = np.full(nprocs,False)
            if nprocs>0:
                Done_in[0] = True
            while Done_in.min()==0:
                if rank >= min(nprocs-1,1) and not Done_in[rank]:
                    rank_now = rank
                    req = comm.isend(rank_now,tag=0,dest=0)
                    req.wait()
                if rank ==0:
                    req = comm.irecv(tag=0,source=MPI.ANY_SOURCE)
                    rank_now = req.wait()
                    if not job_i  <len(sto):
                        Done_in[rank_now] = True
                    #print(job_i,rank_now)
                    req = comm.isend(job_i,tag=1,dest=rank_now)
                    req.wait()
                    req = comm.isend(Done_in,tag=2,dest=rank_now)
                    req.wait()
                    if Done_in.min()>0:
                        for rank_i in jobs:
                             comm.Send((Done_in),tag=3,dest=rank_i)
                    if not Done_in[rank_now]:
                        jobs[rank_now].append(job_i)
                    if job_i  <len(sto):
                        i_stars = job_split[job_i]
                        self.spectra_i = self.spectra[i_stars]
                        comm.Send((self.spectra_i), dest=rank_now, tag=4)
                    job_i += 1
                if rank >= min(nprocs-1,1) and not Done_in[rank]:
                    req = comm.irecv(tag=1,source=0)
                    job_i = req.wait()
                    req = comm.irecv(tag=2,source=0)
                    Done_in = req.wait()
                if rank >= min(nprocs-1,1) and not Done_in[rank]:
                    i_stars = job_split[job_i]
                    self.spectra_i = np.zeros((len(i_stars),len(self.freq)))
                    comm.Recv(self.spectra_i,tag=4,source=0)
                if rank >= min(nprocs-1,1) and Done_in[rank]:
                    comm.Recv(Done_in,tag=3,source=0)
                if rank >= min(nprocs-1,1) and not Done_in[rank]:
            # for rank_now in jobs:
            #     if rank == rank_now:
                    # for job_split_i in jobs[rank]:
                        job_split_i = job_i
                        #time_1 = time.time()
                        star_vel = self.svel[i_stars]
                        initial_pos = self.spos[i_stars]
                        if parallel:
                            final_pos = final_pos_0[i_stars]
                            l_pos = len(final_pos)
                        else:
                            l_pos = l_final_pos
                        #print(1,'Running batch',job_split_i+1,'of',len(job_split),'on processor',rank)
                        # if self.new_stars:
                        #     time.sleep(0.05*rank)
                        #     self.get_stars(subset=i_stars)
                        print('Running batch',job_split_i+1,'of',len(job_split),'on processor',rank)
                        batch += 1

                        #print(len(final_pos),len(initial_pos))
                        sto[job_split_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos,\
                                            star_vel,i_stars,metals,densities,\
                                            np.arange(l_pos),cell_based=cell_based,\
                                            parallel=parallel,offset=0,s_emiss=False)
                            #print(sto[job_split_i]['I'].shape)
                        #print(rank,time.time()-time_1)
            #batch = comm.bcast(batch,root=0)
            if rank ==0:
                print('Group Time:',time.time()-time_2)
            jobs = comm.bcast(jobs,root=0)
            for rank_now_i in jobs:
                    #print(rank_now)
                    for job_split_i in jobs[rank_now_i]:
                            sto[job_split_i] = comm.bcast(sto[job_split_i], root=rank_now_i)
                            if rank ==0:
                                i_range = sto[job_split_i]['i']
                                for i in np.arange(len(i_range)):
                                    if sto[job_split_i]['I'][i].sum()>0:
                                        #print(i,sto[job_split_i]['I'][i].sum())
                                        self.I_f_t[i_range[i]] += sto[job_split_i]['I'][i]
                                        self.Q_f_t[i_range[i]] += sto[job_split_i]['Q'][i]
                                        self.U_f_t[i_range[i]] += sto[job_split_i]['U'][i]
                                        self.V_f_t[i_range[i]] += sto[job_split_i]['V'][i]
                            else:
                                sto[job_split_i] = None
            if rank ==0:
                print('Collate Time:',time.time()-time_2)

            #print(len(self.emission),self.emission.sum())
            sto = {}
        if self.plot_ind >0 or not cell_based:
                final_pos_0 = None
                job_split = None
                j_split = None
                if rank ==0:
                    if parallel:
                        final_pos_0 = self.emiss_pos+i_cells_2
                    else:
                        final_pos_0 = None
                    j_split = np.array_split(np.arange(len(self.emiss_pos)),np.maximum(len(self.emiss_pos)/1000,1))
                j_split,final_pos_0 = comm.bcast((j_split,final_pos_0),root = 0)
                for i_jsplit,j_split_i in enumerate(j_split):
                    if rank ==0:
                        print('Running emission group %s of %s' % (i_jsplit+1,len(j_split)))
                    job_split = np.array_split(j_split_i,np.maximum(len(j_split_i)/50,max(1,nprocs-1)))
                    batch = 1
                    rank_ids = {}
                    len_ranks = 0
                    # for job_split_i in np.arange(len(job_split)):
                    #     rank_ids[job_split_i] = np.array(range(len_ranks,len_ranks+len(job_split[job_split_i])))
                    jobs,sto = job_scheduler_2(np.arange(len(job_split)))
                    batch = comm.bcast(batch,root=0)
                    job_i = 0
                    rank_now = 0
                    count = 0
                    Done_in = np.full(nprocs,False)
                    if nprocs>0:
                        Done_in[0] = True
                    while Done_in.min()==0:
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                            rank_now = rank
                            req = comm.isend(rank_now,tag=0,dest=0)
                            req.wait()
                        if rank ==0:
                            req = comm.irecv(tag=0,source=MPI.ANY_SOURCE)
                            rank_now = req.wait()
                            if not job_i  <len(sto):
                                Done_in[rank_now] = True
                            #print(job_i,rank_now)
                            req = comm.isend(job_i,tag=1,dest=rank_now)
                            req.wait()
                            req = comm.isend(Done_in,tag=2,dest=rank_now)
                            req.wait()
                            if Done_in.min()>0:
                                for rank_i in jobs:
                                     comm.Send((Done_in),tag=3,dest=rank_i)
                            if not Done_in[rank_now]:
                                jobs[rank_now].append(job_i)
                            if job_i  <len(sto):
                                #print(job_split[job_i[0]])
                                self.emission_2 = self.emission[job_split[job_i]]
                                self.emiss_pos_2 = self.emiss_pos[job_split[job_i]]
                                self.emiss_vel_2 = self.emiss_vel[job_split[job_i]]
                                comm.Send((self.emission_2), dest=rank_now, tag=4)
                                comm.Send((self.emiss_pos_2), dest=rank_now, tag=5)
                                comm.Send((self.emiss_vel_2), dest=rank_now, tag=6)
                            job_i += 1
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                            req = comm.irecv(tag=1,source=0)
                            job_i = req.wait()
                            req = comm.irecv(tag=2,source=0)
                            Done_in = req.wait()
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                            len_now = len(job_split[job_i])
                            self.emission_2 = np.zeros((len_now,4,len(self.nu)))
                            self.emiss_pos_2 = np.zeros((len_now,3))
                            self.emiss_vel_2 = np.zeros((len_now,3))
                            comm.Recv(self.emission_2,tag=4,source=0)
                            comm.Recv(self.emission_2,tag=5,source=0)
                            comm.Recv(self.emission_2,tag=6,source=0)
                        if rank >= min(nprocs-1,1) and Done_in[rank]:
                            comm.Recv(Done_in,tag=3,source=0)
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                    # for rank_now in jobs:
                    #     if rank == rank_now:
                            # for job_split_i in jobs[rank]:
                                job_split_i = job_i
                                i_stars = job_split[job_split_i]
                                # self.emiss_ind = rank_ids[job_split_i]
                                # star_vel = self.emiss_vel_2
                                # initial_pos = self.emiss_pos_2
                                if parallel:
                                    final_pos = final_pos_0[i_stars]
                                    l_pos = len(final_pos)
                                else:
                                    l_pos = l_final_pos
                                #print(final_pos)
                                print('Running emission batch',job_split_i+1,'of',len(sto),'on processor',rank)
                                batch += 1
                                sto[job_split_i] = self.ray_trace_3_new(ll,ur,self.emiss_pos_2,final_pos,\
                                                    self.emiss_vel_2,i_stars,metals,densities,\
                                                    np.arange(l_pos),offset=self.offset+self.offset_2,cell_based=cell_based,\
                                                        s_emiss=True,parallel=parallel)
                    jobs = comm.bcast(jobs,root=0)
                    #print(rank,jobs)
                    for rank_now_i in jobs:
                            for job_split_i in jobs[rank_now_i]:
                                sto[job_split_i] = comm.bcast(sto[job_split_i], root=rank_now_i)
                                if rank ==0:
                                    i_range = sto[job_split_i]['i']
                                    for i in np.arange(len(i_range)):
                                        if sto[job_split_i]['I'][i].sum()>0:
                                            #print(i_range[i])
                                            #print(i,sto[job_split_i]['I'][i].sum())
                                            self.I_f_t[i_range[i]] += sto[job_split_i]['I'][i]
                                            self.Q_f_t[i_range[i]] += sto[job_split_i]['Q'][i]
                                            self.U_f_t[i_range[i]] += sto[job_split_i]['U'][i]
                                            self.V_f_t[i_range[i]] += sto[job_split_i]['V'][i]
                                else:
                                    sto[job_split_i] = None
                    # if parallel:
                    #     self.offset_2 += len(j_split_i)
                        #print(i,self.I_f_t.sum())
        #print(len(i_stars),'Done')

    def ray_trace_3_new(self,ll,ur,initial_pos,final_pos,star_vel,\
        i_stars,metals,densities,l_final_pos,\
        cell_based=True,s_emiss=False,offset=0,parallel=False):
        time1 = time.time()
        self.time_0 = np.array([time.time()])
        dr,ray_ind = self.ray_trace_1(ll,ur,initial_pos,final_pos,\
                    cell_based=cell_based,parallel=parallel)
        time1 = time.time()-time1
        #print(rank,'ray_box',time1)
        sto_i = {}
        if cell_based:
            dx = (ur[self.cells]-ll[self.cells]).mean(axis=1)
            #print(dx.shape,l_final_pos.shape)
        else:
            dx = 0
        if parallel:
            sto_i['i'] = i_stars+offset
        if not parallel:
            sto_i['i'] = np.arange(len(l_final_pos))
        sto_i['I'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['Q'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['U'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['V'] = np.zeros((len(l_final_pos),len(self.nu)))
        self.time_0 = np.append(self.time_0,time.time())
        tau_i_j,bigcount = self.ray_trace_4(final_pos,initial_pos,ray_ind,dr,densities,metals,star_vel)
        self.time_0 = np.append(self.time_0,time.time())
        sto_i = self.ray_trace_5(sto_i,final_pos,initial_pos,dx,dr,ray_ind,i_stars,star_vel,\
                tau_i_j,cell_based=cell_based,s_emiss=s_emiss)
        self.time_0 = np.append(self.time_0,time.time())
        self.time_0 = np.diff(self.time_0)
        print(rank,self.time_0/self.time_0.sum(),self.time_0.sum(),self.time_0.sum()/bigcount)
        return sto_i



    def ray_trace_3(self,ll,ur,initial_pos,final_pos,star_vel,\
            i_stars,metals,densities,l_final_pos,\
            cell_based=True,s_emiss=False,offset=0,parallel=False):
        time1 = time.time()
        dr,ray_ind = self.ray_trace_1(ll,ur,initial_pos,final_pos,\
                    cell_based=cell_based,parallel=parallel)
        time1 = time.time()-time1
        print(rank,'ray_box',time1)
        sto_i = {}
        if cell_based:
            dx = (ur[self.cells]-ll[self.cells]).mean(axis=1)
            #print(dx.shape,l_final_pos.shape)
        if parallel:
            sto_i['i'] = i_stars+offset
        if not parallel:
            sto_i['i'] = np.arange(len(l_final_pos))
        sto_i['I'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['Q'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['U'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['V'] = np.zeros((len(l_final_pos),len(self.nu)))
        for i in range(len(l_final_pos)):
            self.time_1 = np.array([time.time()])
            ind_count = 0
            if i < len(final_pos):
                dist = np.linalg.norm(final_pos[i]-initial_pos,axis=1)
            else:
                dist = np.zeros(len(initial_pos))
            #print(i,rank,(ray_ind[:,0]==i).sum())
            for j in range(len(initial_pos)):
                if dist[j] >1:
                    bool_i_j = (ray_ind[:,0]==i)*(ray_ind[:,2]==j)
                    if  bool_i_j.sum()>0:
                        if i==0 and j==2:
                            self.time_0 = np.array([time.time()])
                        if not parallel:
                            bool_i_j = (ray_ind[:,0]==i)*(ray_ind[:,2]==j)
                            ind = ray_ind[bool_i_j][:,1]
                            #save_ind = i
                        else:
                            bool_i_j = ray_ind[:,0]==j
                            ind = ray_ind[bool_i_j][:,1]
                            #save_ind = i_stars[i]+offset
                            #print(i_stars[i],offset)
                        dr_i_j = dr[bool_i_j]
                        if dr_i_j.sum() >0 and not parallel:
                            mod = 4*np.pi*dr_i_j.sum()**2
                        elif dr_i_j.sum() >0:
                            mod = 4*np.pi*dist[j]**2
                        else:
                            mod = 4/(6*dx[j]**2)
                        Z = metals[ind][:,np.newaxis]
                        DGRm = mH*10**(2.445*np.log10(Z)-2.029)
                        v_op = star_vel[j][np.newaxis,:]-self.vel[ind]
                        OP =  initial_pos[j]-final_pos[i]
                        v_op_norm = np.linalg.norm(v_op,axis=1)[:,np.newaxis]
                        OP_norm = np.linalg.norm(OP)
                        v_dot = np.dot(v_op,(OP/OP_norm))
                        v_proj = (OP/OP_norm)*v_dot[:,np.newaxis]
                        red = (c_cgs - np.sign(v_dot)*np.linalg.norm(v_proj,axis=1))/c_cgs
                        #
                        ind_j = self.i_temp[ind]
                        if i==0 and j==2:
                            self.time_0 = np.append(self.time_0,time.time())
                        chix = np.zeros((len(ind),len(self.nu)))
                        chix += self.chisdust_0[ind_j]*DGRm + self.chishe[ind_j] + Z*self.chismet_0[ind_j] +\
                                self.chivdust_0[ind_j]*DGRm + self.chivhe[ind_j] + Z*self.chivmet_0[ind_j]
                        # chix = sum_up(self.chisdust_0,self.chismet_0,self.chishe,\
                        #                     self.chivdust_0,self.chivmet_0,self.chivhe,ind_j,Z,DGRm)
                        if i==0 and j==2:
                            self.time_0 = np.append(self.time_0,time.time())
                        #print((self.nu*red[:,np.newaxis]).shape,chix.shape)
                        chix = np.array([np.interp(self.nu,self.nu*red[x],chix[x]) for x in range(len(chix))])
                        extinct = np.exp(-np.sum((dr_i_j*densities[ind])[:,np.newaxis]*chix\
                                                     /mH,axis=0))
                        chix = 0
                        if cell_based:
                            v_op = star_vel[j]-self.vel[self.cells[i]]
                        else:
                            v_op = self.halo_v
                        OP =  initial_pos[j]-final_pos[i]
                        v_op_norm = np.linalg.norm(v_op)
                        OP_norm = np.linalg.norm(OP)
                        v_dot = np.dot(v_op,OP/OP_norm)
                        v_proj = (OP/OP_norm)*v_dot
                        red = (c_cgs - np.sign(v_dot)*np.linalg.norm(v_proj))/c_cgs
                        if dr_i_j.sum() ==0 or len(extinct)==0:
                            extinct = 1
                        if s_emiss:
                            Spectra_j = self.emission_2[j,:]*extinct
                        else:
                            if self.new_stars:
                                self.expand_spectra(self.spectra[j],i_stars[j])
                                Spectra_j = self.Spectra*extinct
                            else:
                                Spectra_j = self.Spectra[i_stars[j],:]*extinct
                        Spectra_j = np.array([np.interp(self.nu,self.nu*red,Spectra_j[x]) for x in range(len(Spectra_j))])
                        sto_i['I'][i] += Spectra_j[0]/mod
                        sto_i['Q'][i] += Spectra_j[1]/mod
                        sto_i['U'][i] += Spectra_j[2]/mod
                        sto_i['V'][i] += Spectra_j[3]/mod
                        if i==0 and j==2:
                            self.time_0 = np.append(self.time_0,time.time())
                            self.time_0 = np.diff(self.time_0)
                            print(rank,self.time_0/self.time_0.sum(),self.time_0.sum(),len(ind))
                        self.time_1 = np.append(self.time_1,time.time())
                        ind_count += len(ind)
            self.time_1 = np.diff(self.time_1)
            print(rank,i,self.time_1.sum(),ind_count,self.time_1.sum()/ind_count)
        return sto_i

    def ray_trace_4(self,final_pos,initial_pos,ray_ind,dr,densities,metals,star_vel):
        tau_i_j = np.zeros((len(final_pos),len(initial_pos),len(self.nu)))
        red = self.redshift(initial_pos,final_pos,ray_ind,star_vel)
        split_inds = np.array_split(np.arange(len(ray_ind)),max(len(ray_ind)/500,1))
        bigcount = 0
        for ind_i,ind in enumerate(split_inds):
            ind_range = np.arange(len(ind))
            tau = np.zeros((len(ind_range),len(self.nu)))
            count = 0
            timei = time.time()
            for j in np.unique(ray_ind[ind,1]):
                bool_j = ray_ind[ind,1] ==j
                Z = metals[j]
                DGRm = mH*10**(2.445*np.log10(Z)-2.029)
                temp_j = self.i_temp[j]
                chix = self.chisdust_0[temp_j]*DGRm + self.chishe[temp_j] + Z*self.chismet_0[temp_j] +\
                                            self.chivdust_0[temp_j]*DGRm + self.chivhe[temp_j] + Z*self.chivmet_0[temp_j]
                chix_arr = np.array([cp.interp(self.nu,self.nu*red[x],chix) for x in ind[bool_j]])
                tau[ind_range[bool_j]] = (dr[ind[bool_j]]*densities[j])[:,np.newaxis]*chix_arr/mH
                count += bool_j.sum()
            #timei = time.time()-timei
            # if ind_i%10 ==0 and count >0:
            #     print(rank,ind_i,timei,timei/count)
            #timei = time.time()
            for i in np.unique(ray_ind[ind,0]):
                bool_i = (ray_ind[ind,0] ==i)
                for j in np.unique(ray_ind[ind][bool_i,2]):
                    bool_i_j = (ray_ind[ind,0] ==i)*(ray_ind[ind,2] ==j)
                    tau_i_j[i][j] += cp.sum(tau[bool_i_j],axis=0)
            #timei = time.time()-timei
            tau = 0
            bigcount += count
            # if ind_i%10 ==0 and count >0:
            #     print(rank,ind_i,timei,timei/count)
        tau_i_j = np.exp(-tau_i_j)
        return tau_i_j,bigcount

    def ray_trace_5(self,sto_i,final_pos,initial_pos,dx,dr,ray_ind,i_stars,\
            star_vel,tau_i_j,cell_based=True,s_emiss=False):
        if cell_based:
            dist_arr = np.zeros((len(final_pos),len(initial_pos)))
            for i in range(len(final_pos)):
                for j in range(len(initial_pos)):
                    bool_i_j = (ray_ind[:,0] ==i)*(ray_ind[:,2] ==j)
                    dist_arr[i][j] = dr[bool_i_j].sum()
        else:
            if len(final_pos)>1:
                dist_arr = distance.cdist(final_pos,initial_pos)
            else:
                dist_arr = np.linalg.norm(final_pos[0]-initial_pos,axis=1)
        mod = 1/(4*np.pi*dist_arr**2)
        for i in range(len(final_pos)):
            if len(final_pos)>1:
                modi = mod[i,:]
            else:
                modi = mod
            if cell_based:
                v_op = star_vel-self.vel[self.cells[i]]
            else:
                v_op = star_vel-self.halo_v
            OP = initial_pos-final_pos[i]
            v_op_norm = np.linalg.norm(v_op,axis=1)
            OP_norm = np.linalg.norm(OP)
            v_dot = np.sum(v_op*(OP/OP_norm),axis=1)
            v_proj = (OP/OP_norm)*v_dot[:,np.newaxis]
            red = (c_cgs - np.sign(v_dot)*np.linalg.norm(v_proj,axis=1))/c_cgs
            if s_emiss:
                Spectra_j = np.swapaxes(self.emission_2,0,1)
            else:
                if self.new_stars:
                    self.expand_spectra(self.spectra_i,i_stars)
                    Spectra_j = self.Spectra
                else:
                    Spectra_j = self.Spectra[i_stars,:]
            if cell_based:
                modi[dist_arr[i,:] < dx[i]] = 4/(6*dx[i]**2)
                tau_i_j[i,dist_arr[i,:] < dx[i]] = 1
            Spectra_j = np.array([[np.interp(self.nu,self.nu*red[y],Spectra_j[x][y]) \
                                     for y in range(len(Spectra_j[0]))] \
                                      for x in range(len(Spectra_j))])
            mod_i_j = tau_i_j[i,:]*modi[:,np.newaxis]
            Spectra_j *= mod_i_j
            sto_i['I'][i] += np.sum(Spectra_j[0],axis=0)
            sto_i['Q'][i] += np.sum(Spectra_j[1],axis=0)
            sto_i['U'][i] += np.sum(Spectra_j[2],axis=0)
            sto_i['V'][i] += np.sum(Spectra_j[3],axis=0)
            #print(sto_i['I'].mean())
        return sto_i


    def ray_trace_1(self,ll,ur,initial_pos,final_pos,cell_based=True,parallel=False):
        if parallel:
            dims = 2
            M = final_pos-initial_pos
        else:
            M = (final_pos[:,np.newaxis]-initial_pos)
            dims = 3
        split_ll = cp.array_split(cp.arange(len(ll)),cp.maximum(len(ll)/200,1))
        ray_ind = cp.array([])
        tmin_f = cp.array([])
        tmax_f = cp.array([])
        for split_ll_i in split_ll:
            if dims==3:
                t0 = (ll[split_ll_i,np.newaxis]-initial_pos)/M[:,np.newaxis]
                t1 = (ur[split_ll_i,np.newaxis]-initial_pos)/M[:,np.newaxis]
            elif dims ==2:
                t0 = (ll[split_ll_i,np.newaxis]-initial_pos)/M
                t1 = (ur[split_ll_i,np.newaxis]-initial_pos)/M
            tmin = cp.minimum(t0, t1)
            tmax = cp.maximum(t0, t1)
            t0,t1 = 0,0
            tmin = tmin.max(axis=dims)
            tmax = tmax.min(axis=dims)
            index = cp.arange(tmin.shape[1])
            bool_tmin = (tmin < tmax)*(tmin <1)*(tmax>0)
            if dims == 3:
                target_ind,cell_ind,star_ind = cp.where(bool_tmin)
            else:
                cell_ind,target_ind = cp.where(bool_tmin)
                star_ind = target_ind
            tmin = tmin[bool_tmin]
            tmax = tmax[bool_tmin]
            bool_tmin = 0
            if len(ray_ind) == 0:
                ray_ind = cp.stack((target_ind,split_ll_i[cell_ind],star_ind),axis=1)
            else:
                ray_ind = cp.vstack((ray_ind,cp.stack((target_ind,split_ll_i[cell_ind],star_ind),axis=1)))
            target_ind,cell_ind,star_ind =0,0,0
            tmin_f = cp.append(tmin_f,tmin)
            tmax_f = cp.append(tmax_f,tmax)
            tmin,tmax = 0,0
        tmin_f = cp.maximum(tmin_f,0)
        if not cell_based:
            tmax_f = cp.minimum(tmax_f,1)
        if dims ==3:
            p_close = tmin_f[:,np.newaxis]*M[ray_ind[:,0],ray_ind[:,2]]+initial_pos[ray_ind[:,2]]
            p_far = tmax_f[:,np.newaxis]*M[ray_ind[:,0],ray_ind[:,2]]+initial_pos[ray_ind[:,2]]
        elif dims==2:
            p_close = tmin_f[:,np.newaxis]*M[ray_ind[:,0]]+initial_pos[ray_ind[:,2]]
            p_far = tmax_f[:,np.newaxis]*M[ray_ind[:,0]]+initial_pos[ray_ind[:,2]]
        dr = cp.linalg.norm(p_far-p_close, axis=1)
        return dr,ray_ind

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

    def set_emiss(self,densities,dx):
        self.mod_h1 = densities[self.i_cen]*dx[self.i_cen]**3
        self.mod_em = densities[self.i_cen]*dx[self.i_cen]**3
        tauhe = (self.chivhe[self.i_temp[self.i_cen]])*self.rp*densities[self.i_cen]/mH
        taumet = (self.chivmet[self.i_temp[self.i_cen]])*self.rp*densities[self.i_cen]/mH
        taudust = (self.chivdust[self.i_temp[self.i_cen]])*self.rp*densities[self.i_cen]/mH
        if self.temp[self.i_temp[self.i_cen]]>=self.Tmax:
            self.emiss_dust = np.zeros(len(self.nu))
            nrg_dust = 0
        else:
            nrg_dust = self.I_f*(1-np.exp(-taudust))/self.mod_h1
            nrg_dust = integrate.simpson(nrg_dust,self.nu)
            dust_predicted = 4 * sigma * self.temps[self.i_cen]**4*self.planck_single(self.chivdust_0[self.i_temp[self.i_cen]])
            #print(dust_predicted)
            dust_i = np.minimum(np.searchsorted(np.array(self.dust_nrg),dust_predicted),len(self.dust_nrg)-1)
            #print(dust_i,nrg_dust,self.dust_nrg,self.dust_nrg.shape)
            self.emiss_dust = (nrg_dust/integrate.simpson(self.emiss_dust_0[dust_i],self.nu))*self.emiss_dust_0[dust_i]
        self.emisshe = self.emisshe_0[self.i_temp[self.i_cen]]*self.blackbod(self.nu,self.temps[self.i_cen])
        self.emissmet = self.emissmet_0[self.i_temp[self.i_cen]]*self.blackbod(self.nu,self.temps[self.i_cen])
        nrg_he = self.I_f*(1-np.exp(-tauhe))/self.mod_h1
        nrg_he = integrate.simpson(nrg_he,self.nu)
        nrg_met = self.I_f*(1-np.exp(-taumet))/self.mod_h1
        nrg_met = integrate.simpson(nrg_met,self.nu)
        I_e = integrate.simpson(self.I_f,self.nu)
        #print(self.i_cen,self.cen_index,nrg_he,nrg_met,nrg_dust,I_e)
        self.emisshe = self.emisshe*nrg_he/integrate.simpson(self.emisshe,self.nu)
        self.emissmet = self.emissmet*nrg_met/integrate.simpson(self.emissmet,self.nu)
        self.emiss = self.emisshe+self.emissmet
        # print(self.emiss)
        # self.emiss *= len(temps)/(len(self.emission))
        # print(self.emiss)

        #print(self.emiss)

    def set_absorb(self):
        Z_sun = self.Z
        DGR = 10**(2.445*np.log10(self.Z)-2.029)
        self.chisdust = self.chisdust_0 * DGR * mH
        self.chivdust = self.chivdust_0 * DGR * mH
        self.chivmet = self.chivmet_0 * Z_sun
        self.chismet = self.chismet_0 * Z_sun
        self.chis = self.chishe + self.chismet + self.chisdust
        self.chiv = self.chivhe + self.chivmet + self.chivdust



    def get_gas_rads(self):
        self.chivhe = self.plot['chivhe']
        self.chishe = self.plot['chishe']
        self.chivmet_0 = self.plot['chivmet']
        self.chismet_0 = self.plot['chismet']
        self.emisshe_0 = self.plot['emisshe']/self.nu
        self.emissmet_0 = self.plot['emissmet']/self.nu


    def blackbod(self,nu,temp):
          h = 1.0546e-27
          c = 2.99792458e10
          k = 1.3807e-16
          a = np.ones(temp.shape)
          nu2 = a[...,np.newaxis]*nu
          pre = (2*h*nu2**3)/c**2
          den = np.exp(h*nu2/(k*temp[...,np.newaxis])) - 1
          black = pre/den
          return black

    def planck(self,kappa):
           b_nu = self.blackbod(self.nu,self.temp)
           planck1 = integrate.simpson(b_nu*kappa,self.nu)/integrate.simpson(b_nu,self.nu)
           return planck1

    def blackbod_single(self):
          h = 1.0546e-27
          c = 2.99792458e10
          k = 1.3807e-16
          pre = (2*h*self.nu**3)/c**2
          den = np.exp(h*self.nu/(k*self.temps[self.i_cen])) - 1
          black = pre/den
          return black

    def planck_single(self,kappa):
           b_nu = self.blackbod_single()
           planck1 = integrate.simpson(b_nu*kappa,self.nu)/integrate.simpson(b_nu,self.nu)
           return planck1

    def get_spos(self):
        self.spos,self.svel,self.bool_inside,\
            self.halo_c,self.halo_v,self.halo_r,\
            self.plot,self.nu,self.temp,self.lums = None, None, None, None ,None, None,None, None, None, None
        if rank ==0:
            stars = np.load(self.star_folder+'starlists_2013.npy',allow_pickle=True).tolist()
            self.spos = stars[self.halo][self.timestep]['positions2']*self.ds.length_unit.in_units('cm').v
            self.svel = stars[self.halo][self.timestep]['vels2']
            self.lums = stars[self.halo][self.timestep]['luminosity2']
            plotfile = self.find_plothype_file(stars)
            stars = 0
            halotree = np.load(self.star_folder+'halotree_2013_final.npy',allow_pickle=True).tolist()
            self.halo_c = halotree[self.halo][self.timestep]['Halo_Center']*self.ds.length_unit.in_units('cm').v #np.average(self.spos,axis=0,weights=self.lums)#
            self.halo_v = halotree[self.halo][self.timestep]['Vel_Com']*self.ds.length_unit.in_units('cm').v
            self.halo_r = halotree[self.halo][self.timestep]['Halo_Radius']*self.ds.length_unit.in_units('cm').v #1.5*self.spos.std(axis=0).mean()#
            self.bool_inside = (np.sum(self.spos > self.halo_c-self.halo_r,axis=1)==3)*\
                        (np.sum(self.spos < self.halo_c+self.halo_r,axis=1)==3)
            self.spos = self.spos[self.bool_inside]
            self.svel = self.svel[self.bool_inside]
            self.lums = self.lums[self.bool_inside]
            print(len(self.spos),"Stars")
            halotree = None
            self.plot = np.load(plotfile,allow_pickle=True).tolist()
            self.nu = self.plot['nu']
            self.temp = self.plot['temp']
        self.spos,self.svel,self.halo_c,self.halo_v,self.halo_r,self.plot,\
                self.nu,self.temp,self.bool_inside,self.lums =comm.bcast((self.spos,self.svel,\
                    self.halo_c,self.halo_v,self.halo_r,self.plot,self.nu,self.temp,self.bool_inside,self.lums),root=0)

    def find_plothype_file(self,stars):
        '''
        Returns, for a given halo at a given timestep, the revelevant absorption file. The radiative background changes for halos at different timesteps.
        Parameters:
            halo (str): Halo Number
            starpath (str): Path to the starlist file
            tstep (int): Timestep
            plotpath (str): Path to the directory containing the plothype files
        Returns:
            str: Path to the relevant plothype file
        '''
        centers = np.load(f'{plotpath}/centers.npy',allow_pickle=True).tolist()
        id_cen = np.load(f'{plotpath}/id_cen.npy',allow_pickle=True).tolist()
        Beta = np.log(stars[self.halo][self.timestep]['i-900_uv-1500_b1-1300_b2-3500_L-ion'][3]/\
            stars[self.halo][self.timestep]['i-900_uv-1500_b1-1300_b2-3500_L-ion'][2])/np.log(7/3)
        psi_ion = np.log10(stars[self.halo][self.timestep]['i-900_uv-1500_b1-1300_b2-3500_L-ion'][4]/\
            (2e15*stars[self.halo][self.timestep]['i-900_uv-1500_b1-1300_b2-3500_L-ion'][1]))
        dist = distance.cdist(np.array([Beta,psi_ion])[:,np.newaxis].T,np.array(centers))
        idx = np.argmin(dist,axis=1)[0]
        return f'{plotpath}/plothype{id_cen[idx][0]}_{id_cen[idx][1]}.npy'



    def get_stars(self,subset=[]):
        ages = self.stars['age2']
        creation_times = self.ds.current_time.in_units('Gyr').v - ages
        metallicities = self.stars['met2']
        masses = self.stars['mass2']
        masses = masses[self.bool_inside]
        ages = ages[self.bool_inside]
        creation_times = creation_times[self.bool_inside]
        metallicities = metallicities[self.bool_inside]
        if len(subset) != 0:
            masses = masses[subset]
            ages = ages[subset]
            creation_times = creation_times[subset]
            metallicities = metallicities[subset]
        lums, self.freq, spectra_i =\
            SSP_interpolator(self.path_to_fsps,self.ds,self.path_to_fsps,ages,creation_times,metallicities,masses)
        self.spectra = spectra_i * 3.828e33

    def expand_spectra(self,spectra,i_star):
        if spectra.ndim >1:
            len_spec = len(spectra)
            spectra  =  np.array([np.interp(self.nu, self.freq[::-1], spectra[i][::-1]) for i in range(len(spectra))])
        else:
            len_spec = 1
            spectra = np.interp(self.nu, self.freq[::-1], spectra[::-1])
        I = spectra
        if spectra.ndim >1:
            Q = spectra * self.randQ_star[i_star][:,np.newaxis]
            U = spectra * self.randU_star[i_star][:,np.newaxis]
        else:
            Q = spectra * self.randQ_star[i_star]
            U = spectra * self.randU_star[i_star]
        V = np.zeros(spectra.shape)
        if spectra.ndim >1:
            self.Spectra = np.zeros((4,len_spec,len(self.nu)))
        else:
            self.Spectra = np.zeros((4,len(self.nu)))
        self.Spectra[0] = I
        self.Spectra[1] = Q
        self.Spectra[2] = U
        self.Spectra[3] = V

    def get_grid_values(self):
        lu = self.ds.length_unit.in_units('cm').v
        reg = self.ds.region(self.halo_c/lu,(self.halo_c-self.halo_r)/lu,\
                        (self.halo_c+self.halo_r)/lu)
        densities = (reg['HI_Density']+reg['HII_Density']+2*reg['H2I_Density']+2*reg['H2II_Density']).in_units('g/cm**3')
        dx = reg['dx'].in_units('cm')
        temps = reg['temperature'].in_units('K')
        metals = reg['metallicity'].in_units('Zsun')
        #print(metals)
        #print(self.vel_halo)
        x,y,z =reg['x'].in_units('cm'),reg['y'].in_units('cm'),reg['z'].in_units('cm')
        ll = np.vstack([x-dx/2,y-dx/2,z-dx/2]).T
        ur = np.vstack([x+dx/2,y+dx/2,z+dx/2]).T
        v_x = reg['velocity_x'].in_units('cm/s')
        v_y = reg['velocity_y'].in_units('cm/s')
        v_z = reg['velocity_z'].in_units('cm/s')
        v = np.concatenate((v_x,v_y,v_z))
        v = np.reshape(v,(len(v_x),3))
        return densities.v,dx.v,temps.v,metals.v,ll.v,ur.v,v.v


        #print(nrg_met/integrate.simpson(self.emissmet_0,self.nu))


    def edges(self,ll,ur):
        center = (ll+ur)/2
        M = center[self.i_cen]-self.spos
        t0 = (ll[self.i_cen]-self.spos)/M
        t1 = (ur[self.i_cen]-self.spos)/M
        t = np.maximum(t0,t1)
        t = t.min(axis=1)
        edge = self.spos+M*t[:, None]
        return edge

    def Cell_Star_Intensity(self,ll,ur,dx,vel):
        I_star,Q_star,U_star,V_star = self.Spectra
        ray_end = self.edges(ll,ur)
        d_ray_stars = np.linalg.norm(ray_end-self.spos,axis=1)
        d_mod = 4*np.pi*d_ray_stars[:,np.newaxis]**2
        #print(d_ray_stars)
        #rel_v = vel[i_cel]
        #nu_mod = nu*(
        #np.array([np.interp(nu, freq[::-1], spectra_i[i][::-1]) for i in range(len(spectra_i))])
        projected_area = (6*dx**2)/4
        I,Q,U,V = I_star/d_mod,Q_star/d_mod,U_star/d_mod,V_star/d_mod
        return projected_area*I.sum(axis=0),projected_area*Q.sum(axis=0),\
            projected_area*U.sum(axis=0),projected_area*V.sum(axis=0)

    def find_Stokes(self,densities,dx):
        Stokes_i = np.zeros((4,len(self.P2)))
        taud = (self.chisdust[self.i_temp[self.i_cen]])*self.rp*densities[self.i_cen]/mH
        taud = np.maximum(taud,1e-10)
        absorb = 1-np.exp(-taud)
        Stokes_i[0] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb + (self.I_f+self.redistI)*absorb \
                + self.mod_em*(1-self.f_em)*(self.emiss+self.redistem)*absorb
        Stokes_i[1] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb*self.randQ[self.i_cen] + \
                    (self.Q_f+self.redistQ)*absorb
        Stokes_i[2] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb*self.randU[self.i_cen] + (self.U_f+self.redistU)*absorb
        Stokes_i[3] = (1-self.f_em)*self.mod_h1*self.emiss_dust*0*absorb + (self.V_f+self.redistV)*absorb
        Scatter = np.zeros((len(self.P2),4,4))
        Scatter[:,0,0] = self.P1
        Scatter[:,1,1] = self.P1
        Scatter[:,0,1] = self.P2
        Scatter[:,1,0] = self.P2
        Scatter[:,2,2] = self.P3
        Scatter[:,3,3] = self.P3
        Scatter[:,2,3] = -self.P4
        Scatter[:,3,2] = self.P4
        s_0 = Scatter
        Stokes_f = np.zeros((4,len(self.P2)))
        for i in range(min(int(self.Ns_all_dust.max()+1),100)):
            bool_Ns = self.Ns_all_dust>i
            tauv = (self.chiv[self.i_temp[self.i_cen]])*np.minimum(self.dust_lams,dx[self.i_cen])*densities[self.i_cen]/mH
            absorb = np.exp(-tauv)[bool_Ns]
            Stokes_f[:,bool_Ns] = \
                (s_0.T * Stokes_i).sum(axis=1)[:,bool_Ns]*(np.minimum(self.Ns_all_dust[bool_Ns]-i,1))*absorb
            s_0 = np.matmul(Scatter,s_0)
        self.Stokes_f = Stokes_f

    def find_Scattered(self,densities,dx):
        Scattered = np.zeros((4,len(self.P2)))
        tau = (self.chiv[self.i_temp[self.i_cen]])*self.rp*densities[self.i_cen]/mH
        tau = np.maximum(tau,1e-10)
        taud = (self.chisdust[self.i_temp[self.i_cen]])*self.rp*densities[self.i_cen]/mH
        taud = np.maximum(tau,1e-10)
        absorb = np.exp(-taud)*np.exp(-tau)
        #pass_through = np.exp(-tau)
        Scattered[0] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb +\
                self.mod_em*(1-self.f_em)*(self.emiss+self.redistem)*absorb +\
                (1-self.f_em)*self.redistI*absorb
        Scattered[1] = (1-self.f_em)*self.mod_h1*self.emiss_dust*self.randQ[self.i_cen]*absorb +\
               (1-self.f_em)*self.redistQ*absorb
        Scattered[2] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb*self.randU[self.i_cen] +\
               (1-self.f_em)*self.redistU*absorb
        Scattered[3] = (1-self.f_em)*self.mod_h1*self.emiss_dust*0*absorb +\
               (1-self.f_em)*self.redistV*absorb
        self.Scattered = Scattered


    def find_Emitted(self,dx,densities):
        Emitted = np.zeros((4,len(self.P2)))
        d_em = np.minimum(self.lams,dx[self.i_cen])/2
        tau = (self.chiv[self.i_temp[self.i_cen]])*d_em*densities[self.i_cen]/mH
        tau = np.maximum(tau,1e-10)
        absorb = (1-np.exp(-tau))/tau
        Emitted[0] = self.f_em*self.mod_h1*self.emiss_dust*absorb +\
                self.mod_em*(self.f_em)*(self.emiss+self.redistem)*absorb + \
                    (self.f_em)*self.redistI*absorb
        Emitted[1] = self.f_em*self.mod_h1*self.emiss_dust*absorb*self.randQ[self.i_cen] + \
                    (self.f_em)*self.redistQ*absorb
        Emitted[2] = self.f_em*self.mod_h1*self.emiss_dust*absorb*self.randU[self.i_cen] + \
                    (self.f_em)*self.redistU*absorb
        Emitted[3] = self.f_em*self.mod_h1*self.emiss_dust*0*absorb + \
                    (self.f_em)*self.redistV*absorb
        self.Emitted = Emitted
        #print(self.mod_em,self.emiss.sum(),absorb.sum(),tau)


    def find_Atten(self,densities,dx):
        Atten = np.zeros((4,len(self.P2)))
        d_em = np.minimum(self.lams,dx[self.i_cen])/2
        tau = (self.chiv[self.i_temp[self.i_cen]]+self.chis[self.i_temp[self.i_cen]])*dx[self.i_cen]*densities[self.i_cen]/mH
        tau = np.maximum(tau,1e-10)
        absorb = np.exp(-tau)
        Atten[0] = self.I_f*absorb
        Atten[1] = self.Q_f*absorb
        Atten[2] = self.U_f*absorb
        Atten[3] = self.V_f*absorb
        Initial = np.zeros((4,len(self.P2)))
        Initial[0] = self.I_f
        Initial[1] = self.Q_f
        Initial[2] = self.U_f
        Initial[3] = self.V_f
        self.Atten = Atten
        self.Initial = Initial

    def plotting(self,dx):
        wav = c/self.nu
        wav = wav/1e4
        labs = ['I','|Q|','|U|','|V|']
        colors = ['red','green','orange','purple','blue','brown']
        if self.Stokes_f.max() >0:
            for i in range(len(self.Stokes_f)):
                if self.Stokes_f[i].max()>0:
                    plt.plot(wav,np.abs(self.Stokes_f[i]),label='%s Dust Scattered' %labs[i],color=colors[i])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Wavelength [micron]')
            plt.ylabel(r'Emissivity [erg/s/Hz]')
            plt.legend(fontsize='x-small')
            plt.xlim(1e-2,1e3)
            plt.ylim(1e6,1e15)
            #plt.xlim(700,3000)
            #plt.xlim(1200,1230)
            plt.savefig(self.plot_path+'Dust_Scattering_%s_%s.pdf' % (self.i_cen,self.plot_ind))
            plt.clf()



        for i in range(len(self.Emitted)):
            if self.Emitted[i].max()>0:
                plt.plot(wav,np.abs(self.Emitted[i]),label='%s Emitted' %labs[i],color=colors[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Emissivity [erg/s/Hz]')
        plt.legend(fontsize='x-small')
        plt.xlim(1e-2,1e3)
        plt.ylim(1e11,self.Emitted.max()*5)
        plt.savefig(self.plot_path+'Intrinsic_Emission_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()




        for i in range(len(self.Atten)):
            if self.Atten[i].max()>0:
                plt.plot(wav,np.abs(self.Atten[i])/(dx[self.i_cen]**3),label='%s Attenuated' %labs[i],color=colors[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Photon Density [erg/s/cm$^3$/Hz]')
        plt.legend(fontsize='x-small')
        plt.xlim(1e-2,1e3)
        plt.ylim(1e-50,1e-32)
        #plt.xlim(700,3000)
        plt.savefig(self.plot_path+'Attenuated_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()



        for i in range(len(self.Scattered)):
            if self.Scattered[i].max()>0:
                plt.plot(wav,np.abs(self.Scattered[i]),label='%s Scattered Lines' %labs[i],color=colors[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Emissivity [erg/s/Hz]')
        plt.legend(fontsize='x-small')
        plt.xlim(1e-2,1e3)
        plt.ylim(1e4,1e23)
        plt.savefig(self.plot_path+'Scattered_Lines_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()



        tot_emission = (self.Stokes_f+self.Emitted+self.Atten+self.Scattered)/(dx[self.i_cen]**3)
        for i in range(len(tot_emission)):
            if self.Initial[i].max()>0:
                plt.plot(wav,np.abs(self.Initial[i])/(dx[self.i_cen]**3),':',label='%s External' %labs[i],color=colors[i],linewidth=0.3)
            if tot_emission[i].max()>0:
                plt.plot(wav,np.abs(tot_emission[i]),label='%s Processed' %labs[i],color=colors[i],linewidth=0.9)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Photon Density [erg/s/cm$^3$/Hz]')
        plt.legend(fontsize='x-small')
        plt.ylim(1e-42,tot_emission.max()*5)
        plt.xlim(1e-2,1e3)
        plt.savefig(self.plot_path+'Total_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()


        tot_emission = (self.Stokes_f+self.Emitted+self.Atten+self.Scattered)/(dx[self.i_cen]**3)
        for i in range(len(tot_emission)):
            if self.Initial[i].max()>0:
                plt.plot(wav,np.abs(self.Initial[i])/(dx[self.i_cen]**3),':',label='%s External' %labs[i],color=colors[i],linewidth=0.3)
            if tot_emission[i].max()>0:
                plt.plot(wav,np.abs(tot_emission[i]),label='%s Processed' %labs[i],color=colors[i],linewidth=0.9)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Photon Density [erg/s/cm$^3$/Hz]')
        plt.legend(fontsize='x-small')
        plt.ylim(1e-44,1e-33)
        plt.xlim(1e-1,5)
        plt.savefig(self.plot_path+'TotalUV_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()




        tot_emission = self.Stokes_f+self.Emitted+self.Scattered
        for i in range(len(tot_emission)):
            if tot_emission[i].max()>0:
                plt.plot(wav,np.abs(tot_emission[i]),label='%s Emission' %labs[i],color=colors[i],linewidth=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Emissivity [erg/s/Hz]')
        plt.legend(fontsize='x-small')
        plt.ylim(1e8,tot_emission.max()*5)
        #plt.xlim(700,3000)
        plt.savefig(self.plot_path+'Total_Emission_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()

def contained(x,hull):
  A, b = hull.equations[:, :-1], hull.equations[:, -1:]
  # The hull is defined as all points x for which Ax + b <= 0.
  # We compare to a small positive value to account for floating
  # point issues.
  #
  # Assuming x is shape (m, d), output is boolean shape (m,).
  return np.all(np.asarray(x) @ A.T + b.T < eps, axis=-1)

def convert_RGB(I,nu):
    wav = c/nu
    bool_wav = (wav>400)*(wav<700)
    I = I[:,bool_wav]
    wav = wav[bool_wav]
    R,G,B = 0,0,0
    Rmat = np.zeros((bool_wav.sum(),3))
    for i in range(len(wav)):
        X,Y,Z = XYZfunc(wav[i])
        Rmat[i] = [R,G,B] = 2.3706743*X + -0.9000405*Y + -0.4706338*Z,\
                            -0.5138850*X + 1.4253036*Y + 0.0885814*Z, \
                            0.0052982*X + -0.0146949*Y + 1.0093968*Z
    Rmat = np.maximum(Rmat,0)
    I_f = Rmat[np.newaxis,:]*I[...,np.newaxis]
    print(I_f.shape)
    print(I_f.sum(axis=1))
    return I_f.sum(axis=1)

def convert_RGB_2(I,nu,wav_bands):
    wav = c/nu
    wav = wav/1e4
    Final = np.zeros((len(I),3))
    for i in range(len(wav_bands)):
        wav_bool = (wav>wav_bands[i][0])*(wav<wav_bands[i][1])
        Final[:,i] = integrate.simpson(I[:,wav_bool],nu[wav_bool])
    return Final

def RGB(I,l):
      if l>400 and l <700:
        [X,Y,Z] = XYZfunc(l)
        [R,G,B] = [2.3706743*X + -0.9000405*Y + -0.4706338*Z, -0.5138850*X + 1.4253036*Y + 0.0885814*Z, 0.0052982*X + -0.0146949*Y + 1.0093968*Z]
      else:
        [R,G,B] = [0.,0.,0.]
      return max(R,0.)*I,max(G,0.)*I,max(B,0.)*I

def XYZfunc(l):
      a = {}
      B = {}
      g = {}
      d = {}
      xyz = {}
      a['x'] = [0.362,1.056,-0.0065]
      a['y'] = [0.821,0.286]
      a['z'] = [1.217,0.681]
      B['x'] = [442.,599.8,501.1]
      B['y'] = [568.6,530.9]
      B['z'] = [437.,459.]
      g['x'] = [0.0624,0.0264,0.0490]
      g['y'] = [0.0213,0.0613]
      g['z'] = [0.0845,0.0385]
      d['x'] = [0.0374,0.0323,0.0382]
      d['y'] = [0.0247,0.0322]
      d['z'] = [0.0278,0.0725]
      for k in a:
        xyz[k] = 0
        for j in range(len(a[k])):
           xi = l-B[k][j]
           if xi < 0:
             xyz[k] = xyz[k] + a[k][j]*np.exp(-1.5*((l-B[k][j])*g[k][j])**2)
           else:
             xyz[k] = xyz[k] + a[k][j]*np.exp(-1.5*((l-B[k][j])*d[k][j])**2)
      return xyz['x'],xyz['y'],xyz['z']

def job_scheduler(out_list,ranklim=1e99):
    '''
    Function to schedule jobs for each rank. This is the implementation of MPI to run parallel loops. Works with any given list.
    Parameters:
        out_list (list): List of jobs to be done
    Returns:
        tuple: Dictionary of jobs for each rank, and a dictionary to store the results
    '''
    ranks = np.arange(min(nprocs,ranklim)).astype(int)
    jobs = {i.item(): [] for i in ranks}
    sto = {t: {} for t in out_list}
    if rank == 0:
        count = 0
        while count < len(out_list):
            out_list_2 = np.copy(ranks)
            np.random.shuffle(out_list_2)
            for o in ranks:
                if count + out_list_2[o] < len(out_list):
                    i = count + out_list_2[o].item()
                    jobs[o].append(out_list[i])
            count += len(ranks)
        for o in jobs:
            np.random.shuffle(jobs[o])
    jobs = comm.bcast(jobs, root=0)
    return jobs, sto

def job_scheduler_2(out_list,ranklim=1e99):
    ranks = np.arange(min(nprocs,ranklim)).astype(int)
    #print(ranks)
    jobs = {i.item(): [] for i in ranks}
    sto = {t: {} for t in out_list}
    return jobs, sto

if __name__ == "__main__":
        halo_version = 2013
        test_num = sys.argv[1]
        savestring = sys.argv[2]
        delta = False
        cuda = False
        if cuda:
            import cupy as cp
        else:
            import numpy as cp
        if delta:
            plotpath = '/work/hdd/bezm/gtg115x/Analysis/MassAC'
            ds_path_0 = '/work/hdd/bezm/gtg115x/TreesandLists/'
        else:
            plotpath = 'MassAC'
            ds_path_0 = '/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/'
        c = 2.998e+18
        mH = 1.67e-24
        pc = 3.086e+18
        sigma = 5.67051e-5
        c_cgs =  2.99792458e10
        eps = np.finfo(np.float32).eps
        Radiative_Transfer('7',3)
