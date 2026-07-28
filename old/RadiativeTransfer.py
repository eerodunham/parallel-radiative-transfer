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
import itertools
import periodictable as pt
from scipy.special import kn
# import tracemalloc
# import linecache

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


class Radiative_Transfer():
    def __init__(self,halo,timestep):
      self.run_rounds = 1
      self.plot_path = savestring+'/Results_%s_%s/test_absorb_%s/' % (halo,timestep,test_num)
      self.plot_path2 = savestring+'/Results_%s_%s/' % (halo,timestep)
      if redo_fig or not os.path.exists(self.plot_path+'Final_%s.pdf' % (self.run_rounds-1)):
        self.halo = halo
        self.timestep = timestep
        if delta:
            self.path_to_fsps = '/work/hdd/bezm/gtg115x/Analysis/fsps/'#
            self.gather = False
            self.absorb_path = '/work/hdd/bezm/gtg115x/Make_Absorb/'
        else:
            self.path_to_fsps = '/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/fsps/'
            self.absorb_path = './'
            self.gather = True
        if rank==0:
            if not os.path.exists(self.plot_path):
                os.makedirs(self.plot_path)
        #ds_path = '/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/box_3_z_1/DD0683/output_0683'
        self.Tmax = 1100
        self.plotfile = None
        self.plotfile_0 = None
        self.freq = None
        self.gather = False
        self.plot_ind = 0
        self.root_ranks = np.arange(2) #np.arange(max(int(nprocs)/5,3)).astype(int)
        self.plot_t = {}
        self.star_folder = ds_path_0+savestring+'/'
        ds_path_1 = self.star_folder+'/'+'pfs_allsnaps_%s.txt' % halo_version
        file_list = np.loadtxt(ds_path_1,dtype=str)[:,0]
        self.ds = yt.load(file_list[timestep])
        self.get_spos()
        densities,dx,self.temps,metals,ll,ur,self.vel,h1den,self.elect_fract = self.get_grid_values()
        #metals *= 50
        self.dxmin = dx.min()
        #self.lums = None
        self.or_root = self.root_ranks.max()+1
        if rank in self.root_ranks and rank != self.or_root:
            self.stars = np.load(self.star_folder+'starlists_2013.npy',allow_pickle=True).tolist()[halo][timestep]
            self.get_stars()
            self.stars = 0
        if not os.path.exists(self.plot_path2+'plotfiles.txt'):
            self.plotfile_0 = self.find_initial_files(h1den,densities,dx,ll,ur)
            if rank ==0:
                np.savetxt(self.plot_path2+'plotfiles.txt',self.plotfile_0,fmt='%s')
        else:
            if rank ==0:
                self.plotfile_0 = np.loadtxt(self.plot_path2+'plotfiles.txt',dtype=str)
        if rank ==0:
            print(np.unique(self.plotfile_0))
            self.plotfile_0 = self.plotfile_0.astype('U%s' % (len(plotpath)+len('/plothype_00_00_00.npy')))
        if not delta:
            if rank ==0:
                self.plotfile_0 = np.full(len(self.plotfile_0),'Cloudy/plothype_0_0_0.npy')
                self.plotfile_0 = self.plotfile_0.astype('U%s' % (len(plotpath)+len('/plothype_00_00_00.npy')))
        # if rank==0:
        #     print(self.plotfile_0)
        self.plotfile = None
        (self.plotfile,self.freq) = comm.bcast((self.plotfile_0,self.freq),root=0)
        self.Spectra = np.array([])
        self.plotfile = self.plotfile.astype('U%s' % (len(plotpath)+len('/plothype_00_00_00.npy')))
        np.random.seed(seed=10)
        self.randQ_star = np.random.normal(loc=0.01, scale=0.01, size=len(self.spos))
        self.randU_star = np.random.normal(loc=0.01, scale=0.01, size=len(self.spos))
        ind_temps = np.arange(len(self.temps))[self.temps>1e5]
        self.temp_ind = np.arange(len(self.plot['temp']))
        self.get_gas_rads(emiss=False)
        self.i_temp = np.minimum(np.searchsorted(self.plot['temp'],self.temps),len(self.temp)-1)
        self.load_dust()
        self.deletechi()
        # tracemalloc.start()
        self.plot = None
        self.z_now = self.ds.current_redshift
        self.i_cen_range = None
        self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t = None,None,None,None
        if rank ==0:#np.arange(len(dx))#
            if not os.path.exists(self.plot_path+'nu.txt'):
                np.savetxt(self.plot_path+'nu.txt',self.nu)
            if not os.path.exists(self.plot_path+'cells.npy'):
                if delta:
                    self.i_cen_range = np.arange(len(dx))#np.random.choice(np.arange(len(dx)), 50, replace=False)#np.array([1898,1899,1900])
                else:
                    self.i_cen_range = np.random.choice(np.arange(len(dx)), 300, replace=False)
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
                properties['HI_density'] = h1den
                properties['El_fraction'] = self.elect_fract
                #properties['lums'] = self.lums
                np.save(self.plot_path+'properties.npy',properties)
                properties = None
            self.emission = np.zeros((len(self.i_cen_range),4,len(self.nu)))
            self.emiss_pos = np.zeros((len(self.i_cen_range),3))
            self.emiss_vel = np.zeros((len(self.i_cen_range),3))
            if cmb:
                self.cmb_op = np.zeros((len(self.i_cen_range),3))
        self.i_cen_range = comm.bcast(self.i_cen_range, root=0)
        np.random.seed(seed=20)
        self.cmb_randQ = np.random.normal(loc=0.0, scale=0.1,size=len(dx))
        self.cmb_randU = np.random.normal(loc=0.0, scale=0.1,size=len(dx))
        self.cmb_randV = np.random.normal(loc=0.0, scale=1e-14,size=len(dx))
        self.cmb = cmb*4*np.pi*self.blackbod(self.nu,np.array((1+self.z_now)*2.73))#/c_cgs
        self.cmbQ = self.cmb*1e-6/((1+self.z_now)*2.73)
        self.cmbU = self.cmb*1e-6/((1+self.z_now)*2.73)
        self.cmbV = 0
        self.randQ = np.random.normal(loc=0.0, scale=0.1,size=len(dx))
        self.randU = np.random.normal(loc=0.0, scale=0.1,size=len(dx))
        self.plot_ind = 0
        if len(self.spos) >0:
            for run_round in range(self.run_rounds):
                self.plot_ind = run_round
                self.cell_split = np.array_split(self.i_cen_range,np.maximum(len(self.i_cen_range)/100,1))
                self.new = False
                if rank ==0:
                    #print(np.unique(self.plotfile))
                    self.emission_1 = np.zeros((len(self.i_cen_range),4,len(self.nu)))
                    self.emiss_pos_1 = np.zeros((len(self.i_cen_range),3))
                    self.emiss_vel_1 = np.zeros((len(self.i_cen_range),3))
                    if cmb:
                        self.cmb_op_1 = np.zeros((len(self.i_cen_range),3))
                if not os.path.exists(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,len(self.cell_split)-1)):
                    if rank==0:
                        print('Running round %s of %s' % (run_round+1,self.run_rounds))
                    self.run_transfer(ll,ur,metals,dx,densities)
                    # snapshot = tracemalloc.take_snapshot()
                    # top_stats = snapshot.statistics('traceback')
                    # for stat_i in [0,1,2]:
                    #     stat = top_stats[stat_i]
                    #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
                    #     for line in stat.traceback.format():
                    #         print(line)
                    if rank ==0:
                        self.emission = np.array(np.copy(self.emission_1))
                        self.emiss_pos = np.array(np.copy(self.emiss_pos_1))
                        self.emiss_vel = np.array(np.copy(self.emiss_vel_1))
                        self.emission_1 = None
                        self.emiss_pos_1 = None
                        self.emiss_vel_1 = None
                elif rank==0:
                    len_emiss = 0
                    self.emission_1 = None
                    self.emiss_pos_1 = None
                    self.emiss_vel_1 = None
                    for cell_round in range(len(self.cell_split)):
                        indicies = np.arange(len(self.cell_split[cell_round]))+len_emiss
                        len_emiss += len(self.cell_split[cell_round])
                        self.emission[indicies] = np.load(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round) , allow_pickle=True)
                        self.emiss_pos[indicies] = np.load(self.plot_path+'emiss_pos_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                        self.emiss_vel[indicies] = np.load(self.plot_path+'emiss_vel_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                        if cmb:
                            self.cmb_op[indicies] = np.load(self.plot_path+'cmb_abs_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                if self.plot_ind ==0 and cmb and rank==0:
                    np.save(self.plot_path+'cmb_abs.npy',self.cmb_op)
                test = None
                for rank_i in np.arange(nprocs):
                    test = comm.bcast(test,root=rank_i)
                if redo_fig or not os.path.exists(self.plot_path+'Final_%s.pdf' % self.plot_ind):
                    if rank ==0:
                         #print(self.cmb_op)
                         #print(len(dx)/(len(self.emission)))
                         #print(np.sum(integrate.simpson(self.emission[:,0,:],self.nu))/3e33)
                         #self.emission *= len(dx)/(len(self.emission))
                         print(np.sum(integrate.simpson(self.emission[:,0,:],self.nu))/3e33)
                    self.lum_dist = self.lumdist()
                    self.plotfile = comm.bcast(self.plotfile_0,root=0)
                    self.set_observer(ll,ur,metals,densities,self.halo_c\
                                    +np.array([0,1,0])*self.lum_dist)
                if run_round < self.run_rounds -1:
                    if rank==0:
                        len_emiss = 0
                        for cell_round in range(len(self.cell_split)):
                            indicies = np.arange(len(self.cell_split[cell_round]))+len_emiss
                            len_emiss += len(self.cell_split[cell_round])
                            I_all = np.load(self.plot_path+'I_cell_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                            self.plotfile[self.cell_split[cell_round]] = self.find_new_files(I_all,densities[indicies],self.nu)
                            if not delta:
                                self.plotfile[self.cell_split[cell_round]] = np.full(len(self.cell_split[cell_round]),'Cloudy/plothype_0_0_0.npy')
                            I_all = None
                        self.plotfile_0 = self.plotfile
            if rank==0:
                print(np.unique(self.plotfile))
            self.plotfile = comm.bcast(self.plotfile_0,root=0)
        # if not os.path.exists(self.plot_path+'image_%s.pdf' % self.plot_ind):
        #     self.make_image(ll,ur,np.array([0,1,0])*self.halo_r,metals,densities)


    def run_transfer(self,ll,ur,metals,dx,densities):
            len_emiss = 0
            for cell_round in range(len(self.cell_split)):
                indicies = np.arange(len(self.cell_split[cell_round]))+len_emiss
                i_cells_2 = self.cell_split[cell_round]
                if not os.path.exists(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round)):
                    if rank ==0:
                        print('Running cell batch %s of %s' % (cell_round+1,len(self.cell_split)))
                        print(i_cells_2,self.i_cen_range)
                    self.plotfile = comm.bcast(self.plotfile_0,root=0)
                    if rank==0:
                        print(np.unique(self.plotfile))
                    #self.send_plot()
                    #self.plot_t = {}
                    if not cuda:
                        self.ray_trace_groups(ll,ur,metals,densities,i_cells_2)#self.ray_trace_2(ll,ur,metals,densities,i_cells_2)
                    else:
                        self.ray_trace_2_single(ll,ur,metals,densities,i_cells_2)
                    # if rank==0:
                    #     print(self.I_f_t.sum(axis=1))
                    if rank ==0:
                        time10 = time.time()
                    self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t = comm.bcast((self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t),root=0)
                    if rank ==0:
                         filenames = self.find_new_files(self.I_f_t,densities[i_cells_2],self.nu)
                         #print(self.plotfile[i_cells_2],filenames)
                         self.plotfile[i_cells_2] = filenames
                         print(np.unique(self.plotfile[i_cells_2]))
                         if not delta:
                             self.plotfile[i_cells_2] = np.full(len(i_cells_2),'Cloudy/plothype_0_0_0.npy')
                        # ind_plot = np.arange(len(self.plotfile))
                        # not_i_cell = np.array(list(set(ind_plot.tolist())-set(i_cells_2.tolist()))).astype(int)
                        # #print(not_i_cell)
                        # self.plotfile[not_i_cell] = self.plotfile[i_cells_2][0]
                    #self.plotfile = comm.bcast(self.plotfile,root=0)
                    #self.send_plot()
                    ranks = np.arange(nprocs)
                    jobs,sto = job_scheduler_2(np.arange(len(i_cells_2)))
                    job_i = 0
                    rank_now = 0
                    count = 0
                    Done = np.full(nprocs,True)
                    Done[:20] = False
                    while not Done[rank]:
                            rank_now,root_now,job_i,Done,time3 = job_organizer(np.arange(1,3),job_i,Done,len(sto),or_root=0)
                            if rank ==root_now:
                                self.i_cen = i_cells_2[job_i]
                                self.plot = np.load(self.plotfile[self.i_cen],allow_pickle=True).tolist()
                                self.temp_ind = self.i_temp[self.i_cen]
                                self.get_gas_rads_all()
                                comm.Send((self.plot2), dest=rank_now, tag=4)
                                self.plot = None
                                self.plot2 = None
                            if rank == rank_now:
                                self.plot2 = np.zeros((6,len(self.nu)))
                                comm.Recv(self.plot2,tag=4,source=root_now)
                                self.get_gas_rads_all_2()
                                self.plot2 = None
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
                                self.find_thomson(densities,dx)
                                self.find_Emitted(densities,dx)
                                self.find_Scattered(densities,dx)
                                self.find_Atten(densities,dx)
                                time_f = time.time()
                                # print(self.i_cen,self.elect_fract[self.i_cen],self.temps[self.i_cen])
                                # print(self.i_cen,\
                                #     'Dust Scattering:',integrate.simpson(self.Stokes_f[0,:],self.nu)/3e33,\
                                #     'Emission:',integrate.simpson(self.Emitted[0,:],self.nu)/3e33,\
                                #     'Gas Scattering:',integrate.simpson(self.Scattered[0,:],self.nu)/3e33,\
                                #     'Thomson Scattering:',integrate.simpson(self.Stokes_t[0,:],self.nu)/3e33,\
                                #     'Temperature:',self.temps[self.i_cen],'Electron Fraction:',self.elect_fract[self.i_cen])
                                print(self.i_cen,np.round(2.445*np.log10(self.Z)-2.029,3),\
                                    np.round(np.log10(self.Z),3),self.temps[self.i_cen],\
                                    self.temps[self.i_cen]<self.Tmax,integrate.simpson(self.dust_count,self.nu)/1e33)
                                if self.Stokes_f[0,:].max()>0 and not delta:
                                    wav = c/self.nu
                                    wav = wav/1e4
                                    bool_wav = (wav >1e-6)
                                    tau = (self.thomson*densities[self.i_cen]*self.rp/mH).max()
                                    # pol_t = np.sqrt(self.Stokes_t[1,:]**2+self.Stokes_t[2,:]**2+self.Stokes_t[3,:]**2)
                                    # pol_g = np.sqrt(self.Stokes_f[1,:]**2+self.Stokes_f[2,:]**2+self.Stokes_f[3,:]**2)
                                    print(self.i_cen,'Thomson to Dust Scattering Ratio:',integrate.simpson(self.Stokes_t[0,:][bool_wav],self.nu[bool_wav])/\
                                        integrate.simpson(self.Stokes_f[0,:][bool_wav],self.nu[bool_wav]),'Max Tau:',tau,\
                                        'Cross Section:',self.thomson,'Temperature:',self.temps[self.i_cen],'Electron Fraction:',\
                                        self.elect_fract[self.i_cen],'Dust Emission:',integrate.simpson(self.dust_count,self.nu)/1e33)
                                sto[self.cen_index]['emission'] = self.Stokes_f+self.Emitted+self.Scattered + self.Stokes_t
                                sto[self.cen_index]['emission_pos'] = (ll[self.i_cen]+ur[self.i_cen])/2
                                sto[self.cen_index]['emission_vel'] = self.vel[self.i_cen]
                                if cmb:
                                    sto[self.cen_index]['cmb_abs'] = self.cmb_abs
                                # if not delta:
                                #   self.plotting(dx)
                                #print(integrate.simpson((self.Stokes_f+self.Emitted+self.Scattered)[0],self.nu)/1e33,self.temps[self.i_cen])
                                time_p = time.time()
                                #print(self.i_cen,time_2-time_0,time_f-time_2,time_p-time_f)
                                self.clean_vars()
                                jobs[rank_now].append(job_i)
                    for rank_i in jobs:
                        jobs[rank_i] = comm.bcast(jobs[rank_i],root=rank_i)
                    for rank_now_i in jobs:
                            for i_cen in jobs[rank_now_i]:
                                sto[i_cen] = comm.bcast(sto[i_cen], root=rank_now_i)
                                if rank ==0:
                                    self.emission_1[indicies[i_cen]] = sto[i_cen]['emission']
                                    self.emiss_pos_1[indicies[i_cen]] = sto[i_cen]['emission_pos']
                                    self.emiss_vel_1[indicies[i_cen]] = sto[i_cen]['emission_vel']
                                    if cmb:
                                        self.cmb_op_1[indicies[i_cen]] = sto[i_cen]['cmb_abs']
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
                        np.savetxt(self.plot_path+'index_%s_%s.txt' % (self.plot_ind,cell_round),i_cells_2)
                        if cmb:
                            np.save(self.plot_path+'cmb_abs_%s_%s.npy' % (self.plot_ind,cell_round),self.cmb_op_1[indicies])
                        print('Radiative Transfer Time:',time.time()-time10)
                else:
                    if rank ==0:
                            self.emission_1[indicies] = np.load(self.plot_path+'emission_%s_%s.npy' % (self.plot_ind,cell_round) , allow_pickle=True)
                            self.emiss_pos_1[indicies] = np.load(self.plot_path+'emiss_pos_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                            self.emiss_vel_1[indicies] = np.load(self.plot_path+'emiss_vel_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                            if cmb:
                                self.cmb_op_1[indicies] = np.load(self.plot_path+'cmb_abs_%s_%s.npy' % (self.plot_ind,cell_round), allow_pickle=True)
                len_emiss += len(self.cell_split[cell_round])
                test = None
                for rank_i in np.arange(nprocs):
                    test = comm.bcast(test,root=rank_i)
                self.I_f_t,self.Q_f_t,self.U_f_t,self.V_f_t = None, None, None, None

    def send_plot(self):
        file_now = np.array(list(self.plot_t.keys()))
        if rank ==0:
            print(np.unique(self.plotfile))
            for file_i in np.unique(self.plotfile):
                if file_i not in file_now:
                    self.plot_t[file_i] = np.load(file_i,allow_pickle=True).tolist()
        for file_i in np.unique(self.plotfile):
            if file_i not in file_now:
                if rank !=0:
                    self.plot_t[file_i] = None
                self.plot_t[file_i] = comm.bcast(self.plot_t[file_i],root=0)
        file_now = comm.bcast(file_now,root=0)

    def set_observer(self,ll,ur,metals,densities,observer_pos,plotnum=0):
        if rank==0:
            #print(self.emission[:,0].mean(axis=1))
            print(np.sum(integrate.simpson(self.emission[:,0,:],self.nu))/3e33)
            wav2 = c/self.nu
            wav = (1+self.z_now)*wav2/1e4
            bool_in = (wav <200) * (wav >50)
            #print(wav)
            conv = 1e23
            conv2 = 1
            self.cmb2 = cmb*4*np.pi*self.blackbod(self.nu/(1+self.z_now),np.array(2.73))
            self.cmbQ2 = self.cmb2*1e-6/(2.73)
            self.cmbU2 = self.cmb2*1e-6/(2.73)
            self.cmbV2 = 0
            labs = ['I','|Q|','|U|','|V|']
            colors = ['red','green','orange','purple','blue','brown']
            self.expand_spectra(self.spectra,np.arange(len(self.spectra)))
            dist_mod = 4*np.pi*np.linalg.norm(self.spos-observer_pos,axis=1)**2
            I,Q,U,V = np.sum(self.Spectra/dist_mod[np.newaxis,:,np.newaxis],axis=1)
            dmod = 4*np.pi*self.lum_dist**2
            iscmb = True
            iscmb3 = True
            iscmb1 = False
            bool_ion = wav2 <911
            #iscmb1 *= 1/dmod
            plt.plot(wav,conv*(conv2*I+self.cmb*iscmb1),':',color=colors[0],label=labs[0]+' Initial',linewidth=0.3)
            plt.plot(wav,conv*np.abs(conv2*Q+self.cmbQ*self.cmb_randQ[0]*iscmb1),':',color=colors[1],label=labs[1]+' Initial',linewidth=0.3)
            plt.plot(wav,conv*np.abs(conv2*U+self.cmbU*self.cmb_randU[0]*iscmb1),':',color=colors[2],label=labs[2]+' Initial',linewidth=0.3)
            plt.plot(wav,conv*np.abs(conv2*V+self.cmbV*self.cmb_randV[0]*iscmb1),':',color=colors[3],label=labs[3]+' Initial',linewidth=0.3)
            print(np.sum(integrate.simpson(self.Spectra[0,:],self.nu))/3e33)
            print(integrate.simpson(I[bool_in],self.nu[bool_in]),\
                integrate.simpson(Q[bool_in],self.nu[bool_in]),integrate.simpson(U[bool_in],self.nu[bool_in]),\
                integrate.simpson(V[bool_in],self.nu[bool_in]))
            self.Spectra = np.array([])
        if redo_fig or not os.path.exists(self.plot_path+'I_%s.npy'  % self.plot_ind):
            self.ray_trace_groups(ll,ur,metals,densities,observer_pos[:,np.newaxis],cell_based=False)
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
            all_I = integrate.simpson(conv2*I+self.cmb,self.nu)
            print(I,bool_ion.sum())
            fract_ion_0 = integrate.simpson(I[bool_ion],self.nu[bool_ion])
            I,Q,U,V = self.I_f_t[0],self.Q_f_t[0],self.U_f_t[0],self.V_f_t[0]
            print(integrate.simpson(I[bool_in],self.nu[bool_in]),integrate.simpson(Q[bool_in],self.nu[bool_in]),\
                integrate.simpson(U[bool_in],self.nu[bool_in]),\
                    integrate.simpson(V[bool_in],self.nu[bool_in]))
            if self.z_now >6:
                print(wav2)
                I[wav2 < 1215.67] = 0
                Q[wav2 < 1215.67] = 0
                U[wav2 < 1215.67] = 0
                V[wav2 < 1215.67] = 0
            all_out = integrate.simpson(conv2*I,self.nu)
            fract_ion_1 = integrate.simpson(I[bool_ion],self.nu[bool_ion])
            print(all_out/all_I,fract_ion_1/fract_ion_0)
            plt.plot(wav,conv*(I*conv2+self.cmb2*iscmb),color=colors[0],label=labs[0]+' Final',linewidth=0.3)
            plt.plot(wav,conv*np.abs(Q*conv2+self.cmbQ2*self.cmb_randQ[0]*iscmb),color=colors[1],label=labs[1]+' Final',linewidth=0.3)
            plt.plot(wav,conv*np.abs(U*conv2+self.cmbU2*self.cmb_randU[0]*iscmb),color=colors[2],label=labs[2]+' Final',linewidth=0.3)
            plt.plot(wav,conv*np.abs(V*conv2+self.cmbV2*self.cmb_randV[0]*iscmb),color=colors[3],label=labs[3]+' Final',linewidth=0.3)
            if cmb and iscmb3:
                plt.plot(wav,conv*self.cmb2,'--',color=colors[0],label='Local CMB',linewidth=0.3)
                plt.plot(wav,conv*np.abs(self.cmbQ2*self.cmb_randQ[0]),'--',color=colors[1],linewidth=0.3)
                plt.plot(wav,conv*np.abs(self.cmbU2*self.cmb_randU[0]),'--',color=colors[2],linewidth=0.3)
                #plt.plot(wav,conv*np.abs(self.cmbV2*self.cmb_randQ[0]*c_cgs),'--',color=colors[3])
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Wavelength [micron]')
            plt.ylabel(r'Flux [Jy]')
            plt.legend(fontsize='x-small')
            ymax = 2e-17*(self.halo_r/self.lum_dist)**2
            plt.ylim(conv*ymax*1e-10,max(ymax,5*conv*(I+(iscmb1 or iscmb or iscmb3)*self.cmb).max()))
            plt.xlim((1+self.z_now)*1e-2,(1+self.z_now)*1e3)
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
            self.halo_r2 = 0.2*self.spos.std(axis=0).mean()#self.halo_r/30
            #print(self.halo_r2,self.spos.mean(axis=0))
            pix_cen = np.linspace(-self.halo_r2+2*self.halo_r2/num_pix,self.halo_r2-2*self.halo_r2/num_pix,num_pix)
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
            plt.imshow(Pixel[:,:,:],origin='lower',interpolation='gaussian')
            plt.axis('off')
            plt.savefig(self.plot_path+'image_%s.pdf' % self.plot_ind,bbox_inches='tight',pad_inches=0)
            plt.clf()
            #np.save(self.plot_path+'Pixel_info.npy',Pixel)

    def lumdist(self):
       om = self.ds.omega_matter
       ol = self.ds.omega_lambda
       hc = self.ds.hubble_constant
       zi = self.z_now
       tH = 1/((3.24077929e-18)*hc)
       Ez = np.sqrt(om*(1+zi)**3.+ol)
       Dist = (1+zi)*c_cgs*tH
       intlist = np.linspace(0,zi,5000)
       H = Dist/(np.sqrt(om*(1+intlist)**3.+ol))
       lumdist = integrate.simpson(H,intlist)
       return lumdist

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
        if cmb:
            #h = 6.6261e-27
            #mod_cmb = (dx[self.i_cen])**3/c_cgs
            mod_cmb = 3*np.pi*(dx[self.i_cen])**2#6*dx[self.i_cen]**2 #dx[self.i_cen]**3
            #print(integrate.simpson(self.cmb*mod_cmb/(h*self.nu*c_cgs),self.nu)/(integrate.simpson(self.I_f/(h*self.nu),self.nu)))
            self.I_f += self.cmb*mod_cmb
            self.Q_f += self.cmbQ*self.cmb_randQ[self.i_cen]*mod_cmb
            self.U_f += self.cmbU*self.cmb_randU[self.i_cen]*mod_cmb
            self.V_f += self.cmbV*self.cmb_randV[self.i_cen]*mod_cmb


    def redistribution(self,densities,dx):
        self.redistem = np.zeros(len(self.nu))
        self.redistI = np.zeros(len(self.nu))
        self.redistQ = np.zeros(len(self.nu))
        self.redistU = np.zeros(len(self.nu))
        self.redistV = np.zeros(len(self.nu))
        pe_lim = 0.1
        self.thomson = self.elect_fract[self.i_cen]*6.652458e-25
        peo = np.exp(-self.chis*dx[self.i_cen]*densities[self.i_cen]/mH)
        pe = np.maximum(peo,pe_lim)
        npe = 1 - pe
        Ns = np.arange(200)+1
        Pns = pe*(npe)**(Ns[:,np.newaxis]-1)
        Ns_all = np.sum(Pns*(Ns[:,np.newaxis]-1),axis=0)
        Pns = None
        self.lams = 1/(self.chis*densities[self.i_cen]/mH)
        self.rp = np.minimum(self.lams,dx[self.i_cen])*Ns_all.astype(int)+dx[self.i_cen]
        # self.rp2 = np.copy(self.rp)
        # self.rp2[pe<= pe_lim] = 0
        Ns_all[pe<= pe_lim] = 0
        Ns_all = np.minimum(Ns_all,1)
        pet = np.exp(-self.thomson*dx[self.i_cen]*densities[self.i_cen]/mH)
        self.t_lams = 1/(self.thomson*densities[self.i_cen]/mH)
        Pnt = pet*(1-pet)**(Ns-1)
        self.Net = np.sum(Pnt*(Ns-1),axis=0)
        #print(self.Net)
        self.Ns_all = Ns_all - self.Net
        self.set_emiss(densities,dx)
        groups = np.arange((pe<= pe_lim).sum())
        split_groups = np.array_split(groups,max(len(groups)/50,1))
        self.f_em = np.minimum(1,6*self.lams/dx[self.i_cen])
        for i,group_i in enumerate(split_groups):
            vd = 1.36*np.abs(self.nu[pe<= pe_lim][group_i]-self.nu[pe>.9][:,np.newaxis])
            # taushe = (self.chishe)*self.rp2*densities[self.i_cen]/mH
            # tausmet = (self.chismet)*self.rp2*densities[self.i_cen]/mH
            #print(Ns_all[Ns_all>0.01])
            if len(vd) >0:
                vd = vd.min(axis=0)
                diff_nu = np.abs(self.nu[:,np.newaxis]-self.nu[pe<= pe_lim][group_i])/vd
                vd = None
                #diff_nu[diff_nu>25] = 25
                R = np.sqrt(np.pi)*special.erfc(diff_nu)*np.exp(diff_nu**2)/2
                R[diff_nu>24.9] = 0
                #R[pe<0.2] =0
                diff_nu = None
                R *= peo[:,np.newaxis]
                R = R/R.sum(axis=0)
                #print(rank,i,R.shape)
                self.redistem += (self.emiss[pe<= pe_lim][np.newaxis,group_i]*R).sum(axis=1)
                self.redistI += (self.I_f[pe<= pe_lim][np.newaxis,group_i]*R).sum(axis=1)
                self.redistQ += (self.Q_f[pe<= pe_lim][np.newaxis,group_i]*R).sum(axis=1)
                self.redistU += (self.U_f[pe<= pe_lim][np.newaxis,group_i]*R).sum(axis=1)
                self.redistV += (self.V_f[pe<= pe_lim][np.newaxis,group_i]*R).sum(axis=1)
                R = None
        pe_dust = np.exp(-self.chisdust*dx[self.i_cen]*densities[self.i_cen]/mH)
        Pns = pe_dust*(1-pe_dust)**(Ns[:,np.newaxis]-1)
        self.Ns_all_dust = np.sum(Pns*(Ns[:,np.newaxis]-1),axis=0)
        Pns = None
        self.dust_lams = 1/((self.chisdust+self.thomson)*densities[self.i_cen]/mH)
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
        # norm2 = np.abs(P2).sum(axis=0)/norm
        # norm3 = np.abs(P3).sum(axis=0)/norm
        # norm4 = np.abs(P4).sum(axis=0)/norm
        P1 = P1/norm
        P2 = P2/norm
        P3 = P3/norm
        P4 = P4/norm
        P1_t = P1.sum(axis=0)
        P2_t = P2.sum(axis=0)
        P3_t = P3.sum(axis=0)
        P4_t = P4.sum(axis=0)
        #print(P1_t,P2_t,P3_t,P4_t)
        return P1_t, P2_t, P3_t, P4_t

    def load_dust(self):
        from hyperion.dust import SphericalDust
        d = SphericalDust('hyperion-dust-0.1.0/dust_files/d03_4.0_4.0_A.hdf5')
        nu_d = d.optical_properties.nu
        kdust = np.interp(self.nu,nu_d,d.optical_properties.chi).T
        alb_dust = np.interp(self.nu,nu_d,d.optical_properties.albedo).T
        chisdust = alb_dust*kdust
        chivdust = kdust - chisdust
        bool_dust_nu = d.emissivities.nu >= self.nu.min()
        full = integrate.simpson(np.array(d.emissivities.jnu).T,d.emissivities.nu)
        included = integrate.simpson(np.array(d.emissivities.jnu)[bool_dust_nu,:].T,d.emissivities.nu[bool_dust_nu])
        self.dust_fix = included/full
        self.dust_nu = d.emissivities.nu
        self.emiss_dust_0 = np.array(d.emissivities.jnu)#np.array([np.interp(self.nu, d.emissivities.nu, np.array(d.emissivities.jnu)[:,i]) \
                            #       for i in range(len(d.emissivities.jnu[0]))])
        #print(self.emiss_dust_0.shape)
        self.dust_nrg = np.array(d.emissivities.var)
        self.chisdust_0 = chisdust[np.newaxis,:]
        self.chivdust_0 = chivdust[np.newaxis,:]
        #print(self.chisdust_0.shape,chivdust.shape)
        self.P1, self.P2, self.P3, self.P4 = self.get_scattering(d,nu_d)

    def mean_k_intensity(self,kappa):
        mean = integrate.simpson(self.I_f*kappa,self.nu)/integrate.simpson(self.I_f,self.nu)
        return mean

    def get_star_groups(self,spectra,pos,vel,nus,pos_i,full_spec=True):
        if full_spec:
            if spectra.ndim ==3:
                spectra_t = spectra[:,0,:]
            else:
                spectra_t = spectra
            lum = np.abs(integrate.simpson(spectra_t,nus))
            wav = c/nus
            bool_ion = wav < 911
            #time6 = np.array([time.time()])
            lum_ion = np.abs(integrate.simpson(spectra_t[:,bool_ion],nus[bool_ion]))
        else:
            lum = spectra[:,-2]
            lum_ion = spectra[:,-3]
        #time6 = np.append(time6,time.time())
        bool_threshold,flux_lum = self.get_thresholds(pos,pos_i,lum,lum_ion)
        #time6 = np.append(time6,time.time())
        pos_final,spec_final,vel_final,i_star = self.grouping(spectra,bool_threshold,pos,lum,vel,self.pos_grid,self.Grid,flux_lum)
        #time6 = np.append(time6,time.time())
        #print(np.diff(time6)/np.diff(time6).sum(),time6[-1]-time6[0])
        return pos_final,spec_final,vel_final,i_star


    def get_thresholds(self,centers,pos_i,lum,lum_ion):
        dth1= 5*self.dxmin
        if self.all_pos.ndim >1:
            dist = np.linalg.norm(pos_i - self.all_pos,axis=1)
        else:
            dist = np.linalg.norm(pos_i - self.all_pos)
        flux_ion = self.all_lum_ion/(4*np.pi*dist**2)
        flux_lum = self.all_lum/(4*np.pi*dist**2)
        flux_ion[np.isinf(flux_ion)] = 0
        flux_lum[np.isinf(flux_lum)] = 0
        flux_sort = np.argsort(flux_ion)
        cum_flux = np.cumsum(flux_ion[flux_sort])
        sumflux = np.sum(flux_ion)
        fi2 = flux_ion[flux_sort][cum_flux / sumflux > 1e-3].min()
        fi1 = flux_ion[flux_sort][cum_flux / sumflux > 0.1].min()
        flux_sort = np.argsort(flux_lum)
        cum_flux = np.cumsum(flux_lum[flux_sort])
        sumflux = np.sum(flux_lum)
        ft2 = flux_lum[flux_sort][cum_flux / sumflux > 1e-3].min()
        ft1 = flux_lum[flux_sort][cum_flux / sumflux > 0.1].min()
        if centers.ndim >1 or pos_i.ndim >1:
            dist= np.linalg.norm(pos_i - centers,axis=1)
        else:
            dist= np.linalg.norm(pos_i - centers,axis=1)
        flux_ion = lum_ion/(4*np.pi*dist**2)
        flux_lum = lum/(4*np.pi*dist**2)
        flux_ion[np.isinf(flux_ion)] = 0
        flux_lum[np.isinf(flux_lum)] = 0
        #print(fi1,fi2,ft1,ft2,flux_lum)
        bool_threshold = {}
        bool_threshold[-1] = ((flux_lum >= ft1) + (dist < dth1) + (flux_ion >= fi1)) * (flux_lum!=0)
        bool_threshold[1] = ((flux_lum >= ft2)+ (flux_ion >= fi2)) * (~bool_threshold[-1]) * (flux_lum!=0)
        bool_threshold[0] = (~bool_threshold[-1] & ~bool_threshold[1] ) *(flux_lum !=0)
        #print(bool_threshold,fi1,fi2,ft1,ft2,flux_lum)
        #print(bool_threshold[-1].sum(),bool_threshold[1].sum(),bool_threshold[0].sum())
        #print(bool_threshold)
        return bool_threshold,flux_lum[:,np.newaxis]

    def grouping(self,specs,bool_threshold,pos,lum,vel,pos_grid,Grid,flux_lum):
        spec_final = np.array([])
        pos_final = np.array([])
        vel_final = np.array([])
        i_star = np.array([])
        lum = lum[:,np.newaxis]
        for x in [-1,0,1,]:
            time_9 = time.time()
            ind = np.arange(len(specs))[bool_threshold[x]]
            if x == -1 and len(ind)>0:
                i_star = np.append(i_star,ind)
                if len(pos_final)==0:
                    spec_final = specs[ind]
                    #print(specs[ind])
                    pos_final = pos[ind]
                    vel_final = vel[ind]
                else:
                    spec_final = np.vstack((spec_final,specs[ind]))
                    pos_final = np.vstack((pos_final,pos[ind]))
                    vel_final = np.vstack((vel_final,vel[ind]))
            elif x != -1 and len(ind)>0:
                pos_final_x = np.zeros((len(Grid[x]),3))
                vel_final_x = np.zeros((len(Grid[x]),3))
                spec_final_x = np.zeros((len(Grid[x]),)+specs[0].shape)
                bool_in = np.full(len(Grid[x]),False)
                for y in range(len(Grid[x])):
                    star_ind = pos_grid[x][Grid[x][y][0]][Grid[x][y][1]][Grid[x][y][2]]
                    if len(star_ind) > 0:
                        #print(star_ind,'star')
                        star_ind = star_ind[np.isin(star_ind,ind)]
                        #print(star_ind)
                        if len(star_ind) >0:
                            bool_in[y] = True
                            i_star = np.append(i_star,star_ind[0])
                        if len(star_ind) > 1:
                            #timearray = np.array([time.time()])
                            pos_avg = (pos[star_ind]*flux_lum[star_ind]).sum()/flux_lum[star_ind].sum()
                            #pos_avg = np.average(pos[star_ind],weights=lum[star_ind],axis=0)
                            #timearray = np.append(timearray,time.time())
                            vel_avg = (vel[star_ind]*flux_lum[star_ind]).sum()/flux_lum[star_ind].sum()
                            #vel_avg = np.average(vel[star_ind],weights=lum[star_ind],axis=0)
                            #timearray = np.append(timearray,time.time())
                            spec_avg = specs[star_ind].sum(axis=0)
                            #timearray = np.append(timearray,time.time())
                            pos_final_x[y] += pos_avg
                            vel_final_x[y] += vel_avg
                            spec_final_x[y] += spec_avg
                            #timearray = np.append(timearray,time.time())
                            #print(len(star_ind),np.diff(timearray)/np.diff(timearray).sum(),timearray[-1]-timearray[0])
                        elif len(star_ind) == 1:
                            pos_final_x[y] += pos[star_ind][0]
                            vel_final_x[y] += vel[star_ind][0]
                            spec_final_x[y] += specs[star_ind][0]
                            #timearray = np.append(timearray,time.time())
                            #print(len(star_ind),np.diff(timearray)/np.diff(timearray).sum())
                if len(pos_final)==0:
                    spec_final = spec_final_x[bool_in]
                    pos_final = pos_final_x[bool_in]
                    vel_final = vel_final_x[bool_in]
                else:
                    spec_final = np.vstack((spec_final,spec_final_x[bool_in]))
                    pos_final = np.vstack((pos_final,pos_final_x[bool_in]))
                    vel_final = np.vstack((vel_final,vel_final_x[bool_in]))
            #print(x,time.time()-time_9)
        #print(spec_final.shape,len(i_star),specs)
        #print(spec_final)
        return pos_final,spec_final,vel_final,i_star

    def find_all_lums(self,cell_based=True):
        self.all_lum,self.all_pos,self.all_lum_ion = None, None, None
        if rank ==0:
            self.all_lum = np.abs(integrate.simpson(self.spectra,self.freq))
            self.all_pos = self.spos
            if self.plot_ind >0 or not cell_based:
                self.all_lum = np.append(self.all_lum,np.abs(integrate.simpson(self.emission[:,0,:],self.nu)))
                self.all_pos = np.vstack((self.all_pos,self.emiss_pos))
            wav = c/self.freq
            bool_ion = wav < 911
            self.all_lum_ion = np.abs(integrate.simpson(self.spectra[:,bool_ion],self.freq[bool_ion]))
            if self.plot_ind >0 or not cell_based:
                wav = c/self.nu
                bool_ion = wav < 911
                self.all_lum_ion = np.append(self.all_lum_ion,np.abs(integrate.simpson(self.emission[:,0,bool_ion],self.nu[bool_ion])))
        self.all_lum,self.all_pos,self.all_lum_ion = comm.bcast((self.all_lum,self.all_pos,self.all_lum_ion),root=0)

    def build_hierarchy(self,pos,theta=.5,levels=2,seg=10):

        '''
        Build a multi-level grid and produce per-cell centers-of-mass, masses, a grid index for each particle,
        and boolean adjacency/node masks based on inter-node distances and parent/child occupancy.

        Args:
            pos (ndarray): Particle positions.
            mass (ndarray): Particle masses.
            theta (float): Adjacency threshold.
            levels (int): Number of grid levels.
            seg (int): Number of segments per level.
            Numlength (int): Length of the particle list.

        Returns:
            ll_all (ndarray): The lower-left corner of the entire grid.
            ur_all (ndarray): The upper-right corner of the entire grid.
            CoM (dict): The centers of mass for each grid level.
            Grid (dict): The grid indices for each particle.
            M_total (dict): The total mass for each grid level.
            nodes (dict): The boolean adjacency masks for each grid level.
            P_grid (dict): The grid indices for each particle.
        '''
        grid_res = seg
        pos2 = self.all_pos
        width = pos2.max(axis=0) - pos2.min(axis=0)
        ll_all = (pos2.min(axis=0) - 0.5*width/grid_res) #ll_all = (pos.min(axis=0) - 0.05*width)
        ur_all = (pos2.max(axis=0) + 0.5*width/grid_res) #ur_all = (pos.max(axis=0) + 0.05*width)

        Grid,  P_grid, ll_r, ur_r = {},{},{},{}
        pos_grid = {}
        pos_ind = np.arange(len(pos))
        for r in range(levels): #levels^3
            Grid[r],P_grid[r] = \
                np.empty((0,3), float),np.empty((len(pos),3), int)
            xx,yy,zz = np.meshgrid(np.linspace(ll_all[0], ur_all[0],seg * 2**(r)), \
                np.linspace(ll_all[1], ur_all[1], seg * 2**(r)), np.linspace(ll_all[2], ur_all[2], seg * 2**(r)))

            ll_r[r] = np.concatenate((xx[:-1,:-1,:-1,np.newaxis], yy[:-1,:-1,:-1,np.newaxis], zz[:-1,:-1,:-1,np.newaxis]), axis=3)
            ur_r[r] =  np.concatenate((xx[1:,1:,1:,np.newaxis], yy[1:,1:,1:,np.newaxis], zz[1:,1:,1:,np.newaxis]), axis=3)
            child_grid = list(itertools.product(list(range(seg*2**(r)-1)),repeat=3))
            pos_grid[r] = {}
            for child in child_grid:
                i,j,k = child
                if i not in pos_grid[r]:
                    pos_grid[r][i] = {}
                if j not in pos_grid[r][i]:
                    pos_grid[r][i][j] = {}
                if k not in pos_grid[r][i][j]:
                    pos_grid[r][i][j][k] = {}
                ind = ((pos>ll_r[r][i,j,k])*(pos<=ur_r[r][i,j,k])).all(axis=1)
                pos_grid[r][i][j][k] = pos_ind[ind]
                P_grid[r][ind] = i,j,k
                Grid[r] = np.vstack((Grid[r],list(child)))
        return pos_grid,Grid

    def ray_trace_half_groups(self,ll,ur,metals,densities,i_cells_2,cell_based=True,parallel=False):
        time_2 = time.time()
        final_pos = (ll[i_cells_2]+ur[i_cells_2])/2
        l_final_pos = len(i_cells_2)
        l_final_pos_0 = np.arange(l_final_pos)
        l_final_pos_1 = np.arange(l_final_pos)
        self.offset = 0
        self.cells = i_cells_2
        self.offset_2 = 0
        ranks = np.arange(nprocs)
        self.emiss_ind = []
        self.new_stars = True
        self.find_all_lums()
        if self.gather:
            if rank != self.or_root and rank not in self.root_ranks:
                self.plot_hypes = {}
                for file in np.unique(self.plotfile):
                    self.plot_hypes[file] = np.load(file,allow_pickle=True).tolist()
        if rank ==0:
            self.I_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.Q_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.U_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.V_f_t = np.zeros((l_final_pos,len(self.nu)))
            index = np.arange(len(ll))
        if rank in self.root_ranks:
            self.pos_grid,self.Grid = self.build_hierarchy(self.spos)
        jobs,sto = job_scheduler_2(np.arange(len(i_cells_2)))
        self.i_emiss = np.arange(len(self.spos))
        batch = 1
        job_i = 0
        rank_now = 0
        root_now = 0
        count = 0
        len_stars = 0
        dummy = 1
        count_root = 0
        dummy = comm.bcast(dummy,root=0)
        Done = np.full(nprocs,False)
        while not Done[rank]:
            rank_now,root_now,job_i,Done,time3 = job_organizer(self.root_ranks,job_i,Done,len(sto),or_root=self.or_root)
            if rank ==root_now:
                self.pos_final,self.spectra_i,self.vel_final,i_stars =\
                        self.get_star_groups(self.spectra,\
                        self.spos,self.svel,self.freq,final_pos[job_i])
                len_stars = len(self.pos_final)
                req = comm.isend(len_stars,tag=7,dest=rank_now)
                req.wait()
                comm.Send((i_stars), dest=rank_now, tag=8)
                comm.Send((self.spectra_i), dest=rank_now, tag=4)
                comm.Send((self.pos_final), dest=rank_now, tag=5)
                comm.Send((self.vel_final), dest=rank_now, tag=6)
            if rank == rank_now:
                req = comm.irecv(tag=7,source=root_now)
                len_stars = req.wait()
                self.spectra_i = np.zeros((len_stars,len(self.freq)))
                self.pos_final = np.zeros((len_stars,3))
                self.vel_final = np.zeros((len_stars,3))
                i_stars = np.zeros(len_stars)
                comm.Recv(i_stars,tag=8,source=root_now)
                comm.Recv(self.spectra_i,tag=4,source=root_now)
                comm.Recv(self.pos_final,tag=5,source=root_now)
                comm.Recv(self.vel_final,tag=6,source=root_now)
                i_stars = i_stars.astype(int)
                star_vel = self.vel_final
                initial_pos = self.pos_final
                print('Running batch',job_i+1,'of',len(i_cells_2),'on processor',rank,'Time:',time.time()-time3)
                batch += 1
                l_pos = 1
                sto[job_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos[job_i,np.newaxis],\
                                    star_vel,i_stars,metals,densities,\
                                    np.arange(l_pos),cell_based=cell_based,\
                                    parallel=parallel,offset=0,s_emiss=False)
                jobs[rank_now].append(job_i)
            self.spectra_i = None
            self.pos_final = None
            self.vel_final = None
        for rank_i in jobs:
            jobs[rank_i] = comm.bcast(jobs[rank_i],root=rank_i)
        if rank ==0:
            print('Star Time:',time.time()-time_2)
            self.pos_grid,self.Grid = 0,0
            #print(jobs)
        if rank ==0:
            i_range_t = {}
        Done = True
        i_range = None
        for rank_now_i in jobs:
                #print(rank_now)
                for job_split_i in jobs[rank_now_i]:
                        Done = False
                        Done = comm.bcast(Done, root=0)
                        if rank == rank_now_i:
                            comm.Send((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),dest=0)
                            comm.Send((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),dest=0)
                            comm.Send((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),dest=0)
                            comm.Send((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),dest=0)
                        if rank ==0:
                            sto[job_split_i]['I'] = np.zeros((len(self.nu)))
                            sto[job_split_i]['Q'] = np.zeros((len(self.nu)))
                            sto[job_split_i]['U'] = np.zeros((len(self.nu)))
                            sto[job_split_i]['V'] = np.zeros((len(self.nu)))
                            comm.Recv((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),source=rank_now_i)
                            comm.Recv((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),source=rank_now_i)
                            comm.Recv((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),source=rank_now_i)
                            comm.Recv((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),source=rank_now_i)
                            Done = True
                            self.I_f_t[job_split_i] += sto[job_split_i]['I']
                            self.Q_f_t[job_split_i] += sto[job_split_i]['Q']
                            self.U_f_t[job_split_i] += sto[job_split_i]['U']
                            self.V_f_t[job_split_i] += sto[job_split_i]['V']
                            print(integrate.simpson(sto[job_split_i]['I'],self.nu))
                        Done = comm.bcast(Done, root=0)
        sto = None
        if rank ==0:
            print('Collate Time:',time.time()-time_2)
        if self.plot_ind >0 or not cell_based:
                time_3 = time.time()
                len_emiss = None
                self.root_ranks_e = self.root_ranks#np.arange(min(5,nprocs-2))
                self.or_root_e = self.root_ranks_e.max() + 1
                if rank ==0:
                    len_emiss = len(self.emiss_pos)
                len_emiss = comm.bcast(len_emiss,root=0)
                split_emission = np.array_split(np.arange(len_emiss),nprocs-1)
                i_stars = np.arange(len_emiss)
                for rank_i in np.arange(1,nprocs):
                        if rank ==0:
                            self.emission_2 = self.emission[split_emission[rank_i-1]]
                            comm.Send((self.emission_2), dest=rank_i, tag=rank_i)
                            comm.Send((self.emiss_pos), dest=rank_i, tag=2*rank_i)
                            comm.Send((self.emiss_vel), dest=rank_i, tag=3*rank_i)
                        elif rank == rank_i:
                            self.emission_2 = np.zeros((len(split_emission[rank_i-1]),4,len(self.nu)))
                            self.emiss_pos = np.zeros((len_emiss,3))
                            self.emiss_vel = np.zeros((len_emiss,3))
                            comm.Recv(self.emission_2,tag=rank_i,source=0)
                            comm.Recv(self.emiss_pos,tag=2*rank_i,source=0)
                            comm.Recv(self.emiss_vel,tag=3*rank_i,source=0)
                jobs,sto = job_scheduler_2(np.arange(len(i_cells_2)))
                if rank !=0:
                    for job_i in sto:
                        star_vel = self.emiss_vel[split_emission[rank-1]]
                        initial_pos = self.emiss_pos[split_emission[rank-1]]
                        print('Running batch',job_i+1,'of',len(i_cells_2),'on processor',\
                                rank)
                        l_pos = 1
                        if np.linalg.norm(initial_pos-final_pos[job_i,np.newaxis],axis=1).max() >0:
                            sto[job_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos[job_i,np.newaxis],\
                                                star_vel,i_stars[split_emission[rank-1]],metals,densities,\
                                                np.arange(l_pos),cell_based=cell_based,\
                                                parallel=parallel,offset=0,s_emiss=True)
                            jobs[rank].append(job_i)
                self.emission_2 = None
                if rank != 0:
                    self.emiss_pos = None
                    self.emiss_vel = None
                for rank_i in jobs:
                    jobs[rank_i] = comm.bcast(jobs[rank_i],root=rank_i)
                if rank ==0:
                    print('Emiss Time:',time.time()-time_2)
                    #print(jobs)
                if rank ==0:
                    i_range_t = {}
                Done = True
                i_range = None
                for rank_now_i in jobs:
                        #print(rank_now)
                        for job_split_i in jobs[rank_now_i]:
                                Done = False
                                Done = comm.bcast(Done, root=0)
                                if rank == rank_now_i:
                                    comm.Send((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),dest=0)
                                    comm.Send((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),dest=0)
                                    comm.Send((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),dest=0)
                                    comm.Send((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),dest=0)
                                if rank ==0:
                                    sto[job_split_i]['I'] = np.zeros((len(self.nu)))
                                    sto[job_split_i]['Q'] = np.zeros((len(self.nu)))
                                    sto[job_split_i]['U'] = np.zeros((len(self.nu)))
                                    sto[job_split_i]['V'] = np.zeros((len(self.nu)))
                                    comm.Recv((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),source=rank_now_i)
                                    comm.Recv((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),source=rank_now_i)
                                    comm.Recv((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),source=rank_now_i)
                                    comm.Recv((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),source=rank_now_i)
                                    Done = True
                                    self.I_f_t[job_split_i] += sto[job_split_i]['I']
                                    self.Q_f_t[job_split_i] += sto[job_split_i]['Q']
                                    self.U_f_t[job_split_i] += sto[job_split_i]['U']
                                    self.V_f_t[job_split_i] += sto[job_split_i]['V']
                                    print(integrate.simpson(sto[job_split_i]['I'],self.nu))
                                Done = comm.bcast(Done, root=0)
                sto = {}
                jobs = comm.bcast(jobs,root=0)
                if rank ==0:
                    print('Collate Time:',time.time()-time_2)





    def ray_trace_groups(self,ll,ur,metals,densities,i_cells_2,cell_based=True,parallel=False):
        time_5 = time.time()
        time_2 = time.time()
        time_3 = time.time()
        if cell_based and not parallel:
            final_pos = (ll[i_cells_2]+ur[i_cells_2])/2
            l_final_pos = len(i_cells_2)
            l_final_pos_0 = np.arange(l_final_pos)
            l_final_pos_1 = np.arange(l_final_pos)
            self.cells = i_cells_2
        else:
            final_pos = i_cells_2.T
            l_final_pos = 1
            l_final_pos_0 = np.arange(l_final_pos)
            l_final_pos_1 = np.arange(l_final_pos)
            #print(final_pos)
        self.offset = 0
        self.offset_2 = 0
        ranks = np.arange(nprocs)
        self.emiss_ind = []
        self.new_stars = True
        self.find_all_lums(cell_based=False)
        gather_0 = np.copy(self.gather)
        # if self.plot_ind ==0:
        #     self.gather = True
        if self.gather:
            if rank != self.or_root and rank not in self.root_ranks:
                self.plot_hypes = {}
                for file in np.unique(self.plotfile):
                    self.plot_hypes[file] = np.load(file,allow_pickle=True).tolist()
        if rank ==0:
            self.I_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.Q_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.U_f_t = np.zeros((l_final_pos,len(self.nu)))
            self.V_f_t = np.zeros((l_final_pos,len(self.nu)))
            index = np.arange(len(ll))
        wav2 = c/self.nu
        wav = (1+self.z_now)*wav2/1e4
        self.bool_in = (wav <200) * (wav >50)
        if rank in self.root_ranks:
            self.pos_grid,self.Grid = self.build_hierarchy(self.spos)
            #print(self.spos,self.pos_grid)
        self.i_emiss = np.arange(len(self.spos))
        if cell_based:
            group_size = 5000
        else:
            group_size = 100
        j_split = np.array_split(np.arange(len(self.spos)),max(len(self.spos)/group_size,1))
        for group,j_split_i in enumerate(j_split):
            jobs,sto = job_scheduler_2(np.arange(len(i_cells_2)))
            jobs_2,sto = job_scheduler_2(np.arange(len(final_pos)))
            time_2 = time.time()
            if rank==0:
                print('Running group %s of %s' % (group+1,len(j_split)))
            batch = 1
            job_i = 0
            rank_now = 0
            root_now = 0
            count = 0
            len_stars = 0
            dummy = 1
            count_root = 0
            dummy = comm.bcast(dummy,root=0)
            Done = np.full(nprocs,False)
            while not Done[rank]:
                rank_now,root_now,job_i,Done,time3 = job_organizer(self.root_ranks,job_i,Done,len(final_pos),or_root=self.or_root)
                if rank ==root_now:
                    if cell_based:
                        self.pos_final,self.spectra_i,self.vel_final,i_stars =\
                                self.get_star_groups(self.spectra[j_split_i],\
                                self.spos[j_split_i],self.svel[j_split_i],self.freq,final_pos[job_i])
                    else:
                        self.pos_final = self.spos[j_split_i]
                        self.vel_final = self.svel[j_split_i]
                        self.spectra_i = self.spectra[j_split_i]
                        i_stars = j_split_i
                    len_stars = len(self.pos_final)
                    req = comm.isend(len_stars,tag=7,dest=rank_now)
                    req.wait()
                    i_stars = i_stars.astype(float)
                    #print(i_stars)
                    comm.Send((self.spectra_i), dest=rank_now, tag=4)
                    comm.Send((self.pos_final), dest=rank_now, tag=5)
                    comm.Send((self.vel_final), dest=rank_now, tag=6)
                    comm.Send((i_stars), dest=rank_now, tag=8)
                    root_rank = -1
                if rank == rank_now:
                    req = comm.irecv(tag=7,source=root_now)
                    len_stars = req.wait()
                    i_stars = np.zeros(len_stars)
                    self.spectra_i = np.zeros((len_stars,len(self.freq)))
                    self.pos_final = np.zeros((len_stars,3))
                    self.vel_final = np.zeros((len_stars,3))
                    comm.Recv(self.spectra_i,tag=4,source=root_now)
                    comm.Recv(self.pos_final,tag=5,source=root_now)
                    comm.Recv(self.vel_final,tag=6,source=root_now)
                    comm.Recv(i_stars,tag=8,source=root_now)
                    i_stars = i_stars.astype(int)
                    #print(i_stars,len_stars)
                    #print(self.spectra_i)
                    star_vel = self.vel_final
                    initial_pos = self.pos_final
                    print('Running batch',job_i+1,'of',l_final_pos,'on processor',rank,'Time:',time.time()-time3)
                    l_pos = 1
                    output = {}
                    sto[job_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos[job_i,np.newaxis],\
                                        star_vel,i_stars,metals,densities,\
                                        np.arange(l_pos),cell_based=cell_based,\
                                        parallel=parallel,offset=0,s_emiss=False)
                    jobs[rank].append(job_i)
                    rank_now = -1
                    job_i = None
                self.spectra_i = None
                self.pos_final = None
                self.vel_final = None
            for rank_i in jobs:
                jobs[rank_i] = comm.bcast(jobs[rank_i],root=rank_i)
            if rank ==0:
                print('Star Time:',time.time()-time_2)
                #print(jobs)
            if rank ==0:
                i_range_t = {}
            Done = True
            i_range = None
            for rank_now_i in jobs:
                    #print(rank_now)
                    for job_split_i in jobs[rank_now_i]:
                            Done = False
                            Done = comm.bcast(Done, root=0)
                            if rank == rank_now_i:
                                #print(job_split_i,integrate.simpson(sto[job_split_i]['I'],self.nu))
                                comm.Send((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),dest=0)
                                comm.Send((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),dest=0)
                                comm.Send((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),dest=0)
                                comm.Send((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),dest=0)
                            if rank ==0:
                                sto[job_split_i]['I'] = np.zeros((len(self.nu)))
                                sto[job_split_i]['Q'] = np.zeros((len(self.nu)))
                                sto[job_split_i]['U'] = np.zeros((len(self.nu)))
                                sto[job_split_i]['V'] = np.zeros((len(self.nu)))
                                comm.Recv((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),source=rank_now_i)
                                comm.Recv((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),source=rank_now_i)
                                comm.Recv((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),source=rank_now_i)
                                comm.Recv((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),source=rank_now_i)
                                Done = True
                                self.I_f_t[job_split_i] += sto[job_split_i]['I']
                                self.Q_f_t[job_split_i] += sto[job_split_i]['Q']
                                self.U_f_t[job_split_i] += sto[job_split_i]['U']
                                self.V_f_t[job_split_i] += sto[job_split_i]['V']
                                #print(integrate.simpson(self.I_f_t[0][self.bool_in],self.nu[self.bool_in]))
                            Done = comm.bcast(Done, root=0)
            sto = None
            if rank ==0:
                print('Collate Time:',time.time()-time_2)
        if rank ==0:
            self.pos_grid,self.Grid = 0,0
            print('Total Star Time:',time.time()-time_3)
        if self.plot_ind >0 or not cell_based:
                time_3 = time.time()
                len_emiss = None
                if cell_based:
                    self.root_ranks_e = np.arange(min(5,nprocs-2))#self.root_ranks#np.arange(min(5,nprocs-2))
                else:
                    self.root_ranks_e = np.arange(1)
                self.or_root_e = self.root_ranks_e.max() + 1
                if rank ==0:
                    len_emiss = len(self.emiss_pos)
                len_emiss = comm.bcast(len_emiss,root=0)
                #root_ranks = np.arange(max(int(nprocs)/4,1)).astype(int)
                for rank_i in np.arange(1,len(self.root_ranks_e)):
                        if rank ==0:
                            comm.Send((self.emission), dest=rank_i, tag=rank_i)
                            comm.Send((self.emiss_pos), dest=rank_i, tag=2*rank_i)
                            comm.Send((self.emiss_vel), dest=rank_i, tag=3*rank_i)
                        elif rank == rank_i:
                            self.emission = np.zeros((len_emiss,4,len(self.nu)))
                            self.emiss_pos = np.zeros((len_emiss,3))
                            self.emiss_vel = np.zeros((len_emiss,3))
                            comm.Recv(self.emission,tag=rank_i,source=0)
                            comm.Recv(self.emiss_pos,tag=2*rank_i,source=0)
                            comm.Recv(self.emiss_vel,tag=3*rank_i,source=0)
                final_pos_0 = None
                job_split = None
                if rank in self.root_ranks_e:
                    self.pos_grid,self.Grid = self.build_hierarchy(self.emiss_pos,seg=10)
                #j_split = None
                j_split = np.array_split(np.arange(len_emiss),max(len_emiss/500,1))
                for group,j_split_i in enumerate(j_split):
                    jobs_2,sto = job_scheduler_2(np.arange(len(final_pos)))
                    jobs,sto = job_scheduler_2(np.arange(len(final_pos)))
                    time_2 = time.time()
                    if rank==0:
                        print('Running group %s of %s' % (group+1,len(j_split)))
                    batch = 1
                    job_i = 0
                    rank_now = 0
                    root_now = 0
                    count = 0
                    len_stars = 0
                    count = comm.bcast(count,root=self.or_root_e)
                    Done = np.full(nprocs,True)
                    Done[:8*len(self.root_ranks_e)+4] = False
                    jobs = {i.item(): [] for i in ranks}
                    # Done = np.full(nprocs,False)
                    while not Done[rank]:
                        rank_now,root_now,job_i,Done,time3 = job_organizer(self.root_ranks_e,job_i,Done,len(final_pos),or_root=self.or_root_e)
                        #job_i = j_split_i[min(job_i_0,len(j_split_i)-1)]
                        #print(j_split_i,job_i)
                        if rank == root_now:
                            # self.pos_final,self.emission_f,self.vel_final,i_stars =\
                            #         self.get_star_groups(self.emission[j_split_i],\
                            #         self.emiss_pos[j_split_i],self.emiss_vel[j_split_i],\
                            #         self.nu,final_pos[job_i])
                            i_stars = np.array(np.copy(j_split_i))
                            i_stars = i_stars.astype(float)
                            self.pos_final = self.emiss_pos[j_split_i]
                            self.vel_final = self.emiss_vel[j_split_i]
                            len_stars = len(self.pos_final)
                            req = comm.isend(len_stars,tag=7,dest=rank_now)
                            req.wait()
                            comm.Send((self.pos_final), dest=rank_now, tag=5)
                            comm.Send((self.vel_final), dest=rank_now, tag=6)
                            comm.Send((i_stars), dest=rank_now, tag=8)
                            break_emiss = np.array_split(np.arange(len_stars),np.maximum(len_stars/1000,1))
                            for break_i in break_emiss:
                                #print(self.emission_f.shape,self.emission[j_split_i].shape)
                                self.emission_2 = self.emission[j_split_i][break_i]
                                comm.Send((self.emission_2), dest=rank_now, tag=4)
                            #self.emission_f = None
                            root_rank = -1
                        if rank == rank_now:
                            req = comm.irecv(tag=7,source=root_now)
                            len_stars = req.wait()
                            i_stars = np.zeros(len_stars)
                            self.pos_final = np.zeros((len_stars,3))
                            self.vel_final = np.zeros((len_stars,3))
                            comm.Recv(self.pos_final,tag=5,source=root_now)
                            comm.Recv(self.vel_final,tag=6,source=root_now)
                            comm.Recv(i_stars,tag=8,source=root_now)
                            i_stars = i_stars.astype(int)
                            break_emiss = np.array_split(np.arange(len_stars),np.maximum(len_stars/1000,1))
                            for group_i,break_i in enumerate(break_emiss):
                                self.emission_2 = np.zeros((len(break_i),4,len(self.nu)))
                                comm.Recv(self.emission_2,tag=4,source=root_now)
                                #print(integrate.simpson(self.emission_2,self.nu))
                                star_vel = self.vel_final[break_i]
                                initial_pos = self.pos_final[break_i]
                                print('Running group',group_i+1,'of',len(break_emiss),\
                                    'batch',job_i+1,'of',l_final_pos,'on processor',\
                                        rank,'Time:',time.time()-time3)
                                batch += 1
                                l_pos = 1
                                sto[job_i][group_i] = self.ray_trace_3_new(ll,ur,initial_pos,final_pos[job_i,np.newaxis],\
                                                    star_vel,i_stars[break_i],metals,densities,\
                                                    np.arange(l_pos),cell_based=cell_based,\
                                                    parallel=parallel,offset=0,s_emiss=True)
                                jobs[rank].append([job_i,group_i])
                            rank_now = -1
                            job_i = None
                        self.emission_2 = None
                        self.pos_final = None
                        self.vel_final = None
                    for rank_i in jobs:
                        jobs[rank_i] = comm.bcast(jobs[rank_i],root=rank_i)
                    if rank ==0:
                        print('Emiss Time:',time.time()-time_2)
                        #print(jobs)
                    if rank ==0:
                        i_range_t = {}
                    Done = True
                    i_range = None
                    for rank_now_i in jobs:
                            #print(rank_now)
                            for groups in jobs[rank_now_i]:
                                    Done = False
                                    job_split_i = groups[0]
                                    Done = comm.bcast(Done, root=0)
                                    if rank == rank_now_i:
                                        #print(groups)
                                        #print(sto)
                                        comm.Send((sto[groups[0]][groups[1]]['I']),tag=job_split_i+len(i_cells_2),dest=0)
                                        comm.Send((sto[groups[0]][groups[1]]['Q']),tag=job_split_i+2*len(i_cells_2),dest=0)
                                        comm.Send((sto[groups[0]][groups[1]]['U']),tag=job_split_i+3*len(i_cells_2),dest=0)
                                        comm.Send((sto[groups[0]][groups[1]]['V']),tag=job_split_i+4*len(i_cells_2),dest=0)
                                    if rank ==0:
                                        sto[job_split_i]['I'] = np.zeros((len(self.nu)))
                                        sto[job_split_i]['Q'] = np.zeros((len(self.nu)))
                                        sto[job_split_i]['U'] = np.zeros((len(self.nu)))
                                        sto[job_split_i]['V'] = np.zeros((len(self.nu)))
                                        comm.Recv((sto[job_split_i]['I']),tag=job_split_i+len(i_cells_2),source=rank_now_i)
                                        comm.Recv((sto[job_split_i]['Q']),tag=job_split_i+2*len(i_cells_2),source=rank_now_i)
                                        comm.Recv((sto[job_split_i]['U']),tag=job_split_i+3*len(i_cells_2),source=rank_now_i)
                                        comm.Recv((sto[job_split_i]['V']),tag=job_split_i+4*len(i_cells_2),source=rank_now_i)
                                        Done = True
                                        self.I_f_t[job_split_i] += sto[job_split_i]['I']
                                        self.Q_f_t[job_split_i] += sto[job_split_i]['Q']
                                        self.U_f_t[job_split_i] += sto[job_split_i]['U']
                                        self.V_f_t[job_split_i] += sto[job_split_i]['V']
                                        print(job_split_i,integrate.simpson(self.I_f_t[0][self.bool_in],self.nu[self.bool_in]))
                                        #print(job_split_i,integrate.simpson(self.I_f_t[job_split_i],self.nu))
                                    Done = comm.bcast(Done, root=0)
                    jobs = comm.bcast(jobs,root=0)
                    if rank ==0:
                        print('Collate Time:',time.time()-time_2)
                    sto = None
                if rank in np.arange(1,len(self.root_ranks_e)):
                        self.pos_grid,self.Grid = 0,0
                        self.emission = 0
                        self.emiss_pos = 0
                        self.emiss_vel = 0
                if rank ==0:
                    print('Total Emission Time:',time.time()-time_3)
        if self.gather:
            if rank != self.or_root and rank not in self.root_ranks:
                self.plot_hypes = {}
        self.bool_in = None
        if rank ==0:
            print('Total Time:',time.time()-time_5)




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
                        self.spectra_j = self.spectra[i_stars]
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
        if self.gather:
            self.plot_hypes = {}
            for file in np.unique(self.plotfile):
                self.plot_hypes[file] = np.load(file,allow_pickle=True).tolist()
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
            job_split = np.array_split(j_split_i,np.maximum(len(j_split_i)/80,max(1,nprocs-1)))
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
                        #print(integrate.simpson(self.spectra_i,self.freq))
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
            if rank ==0:
                i_range_t = {}
            Done = True
            i_range = None
            jobs = comm.bcast(jobs,root=0)
            for rank_now_i in jobs:
                    #print(rank_now)
                    for job_split_i in jobs[rank_now_i]:
                            Done = False
                            if 'i' not in sto[job_split_i]:
                                sto[job_split_i]['i'] = None
                            #print(sto[job_split_i])
                            i_range = comm.bcast(sto[job_split_i]['i'], root=rank_now_i)
                            if rank == rank_now_i:
                                comm.Send((sto[job_split_i]['I']),tag=job_split_i+len(j_split),dest=0)
                                comm.Send((sto[job_split_i]['Q']),tag=job_split_i+2*len(j_split),dest=0)
                                comm.Send((sto[job_split_i]['U']),tag=job_split_i+3*len(j_split),dest=0)
                                comm.Send((sto[job_split_i]['V']),tag=job_split_i+4*len(j_split),dest=0)
                            if rank ==0:
                                i_range_t[job_split_i] = i_range
                                sto[job_split_i]['I'] = np.zeros((len(i_range),len(self.nu)))
                                sto[job_split_i]['Q'] = np.zeros((len(i_range),len(self.nu)))
                                sto[job_split_i]['U'] = np.zeros((len(i_range),len(self.nu)))
                                sto[job_split_i]['V'] = np.zeros((len(i_range),len(self.nu)))
                                comm.Recv((sto[job_split_i]['I']),tag=job_split_i+len(j_split),source=rank_now_i)
                                comm.Recv((sto[job_split_i]['Q']),tag=job_split_i+2*len(j_split),source=rank_now_i)
                                comm.Recv((sto[job_split_i]['U']),tag=job_split_i+3*len(j_split),source=rank_now_i)
                                comm.Recv((sto[job_split_i]['V']),tag=job_split_i+4*len(j_split),source=rank_now_i)
                                Done = True
                                #print(integrate.simpson(sto[job_split_i]['I'],self.nu))
                            else:
                                sto[job_split_i] = None
                            Done = comm.bcast(Done, root=0)
            if rank ==0:
                for rank_now_i in jobs:
                        #print(rank_now)
                        for job_split_i in jobs[rank_now_i]:
                            i_range = i_range_t[job_split_i]
                            for i in np.arange(len(i_range)):
                                if sto[job_split_i]['I'][i].sum()>0:
                                    #print(i,sto[job_split_i]['I'][i].sum())
                                    self.I_f_t[i_range[i]] += sto[job_split_i]['I'][i]
                                    self.Q_f_t[i_range[i]] += sto[job_split_i]['Q'][i]
                                    self.U_f_t[i_range[i]] += sto[job_split_i]['U'][i]
                                    self.V_f_t[i_range[i]] += sto[job_split_i]['V'][i]
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
                        time_2 = time.time()
                    job_split = np.array_split(j_split_i,np.maximum(len(j_split_i)/80,max(1,nprocs-1)))
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
                            comm.Recv(self.emiss_pos_2,tag=5,source=0)
                            comm.Recv(self.emiss_vel_2,tag=6,source=0)
                        if rank >= min(nprocs-1,1) and Done_in[rank]:
                            comm.Recv(Done_in,tag=3,source=0)
                        if rank >= min(nprocs-1,1) and not Done_in[rank]:
                    # for rank_now in jobs:
                    #     if rank == rank_now:
                            # for job_split_i in jobs[rank]:
                                job_split_i = job_i
                                i_stars = job_split[job_split_i]
                                #print(integrate.simpson(self.emission_2[:,0],self.nu))
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
                    if rank ==0:
                        print('Group Time:',time.time()-time_2)
                    if rank ==0:
                        i_range_t = {}
                    Done = True
                    i_range = None
                    for rank_now_i in jobs:
                            #print(rank_now)
                            for job_split_i in jobs[rank_now_i]:
                                    Done = False
                                    if 'i' not in sto[job_split_i]:
                                        sto[job_split_i]['i'] = None
                                    #print(sto[job_split_i])
                                    i_range = comm.bcast(sto[job_split_i]['i'], root=rank_now_i)
                                    if rank == rank_now_i:
                                        comm.Send((sto[job_split_i]['I']),tag=job_split_i+len(j_split),dest=0)
                                        comm.Send((sto[job_split_i]['Q']),tag=job_split_i+2*len(j_split),dest=0)
                                        comm.Send((sto[job_split_i]['U']),tag=job_split_i+3*len(j_split),dest=0)
                                        comm.Send((sto[job_split_i]['V']),tag=job_split_i+4*len(j_split),dest=0)
                                    if rank ==0:
                                        i_range_t[job_split_i] = i_range
                                        sto[job_split_i]['I'] = np.zeros((len(i_range),len(self.nu)))
                                        sto[job_split_i]['Q'] = np.zeros((len(i_range),len(self.nu)))
                                        sto[job_split_i]['U'] = np.zeros((len(i_range),len(self.nu)))
                                        sto[job_split_i]['V'] = np.zeros((len(i_range),len(self.nu)))
                                        comm.Recv((sto[job_split_i]['I']),tag=job_split_i+len(j_split),source=rank_now_i)
                                        comm.Recv((sto[job_split_i]['Q']),tag=job_split_i+2*len(j_split),source=rank_now_i)
                                        comm.Recv((sto[job_split_i]['U']),tag=job_split_i+3*len(j_split),source=rank_now_i)
                                        comm.Recv((sto[job_split_i]['V']),tag=job_split_i+4*len(j_split),source=rank_now_i)
                                        Done = True
                                        #print(integrate.simpson(sto[job_split_i]['I'],self.nu))
                                    else:
                                        sto[job_split_i] = None
                                    Done = comm.bcast(Done, root=0)
                    if rank ==0:
                        for rank_now_i in jobs:
                                #print(rank_now)
                                for job_split_i in jobs[rank_now_i]:
                                    i_range = i_range_t[job_split_i]
                                    for i in np.arange(len(i_range)):
                                        if sto[job_split_i]['I'][i].sum()>0:
                                            #print(i,sto[job_split_i]['I'][i].sum())
                                            self.I_f_t[i_range[i]] += sto[job_split_i]['I'][i]
                                            self.Q_f_t[i_range[i]] += sto[job_split_i]['Q'][i]
                                            self.U_f_t[i_range[i]] += sto[job_split_i]['U'][i]
                                            self.V_f_t[i_range[i]] += sto[job_split_i]['V'][i]
                    if rank ==0:
                        print('Collate Time:',time.time()-time_2)
                    # if parallel:
                    #     self.offset_2 += len(j_split_i)
                        #print(i,self.I_f_t.sum())
        #print(len(i_stars),'Done')
        if self.gather:
            self.plot_hypes = {}

    def ray_trace_3_new(self,ll,ur,initial_pos,final_pos,star_vel,\
        i_stars,metals,densities,l_final_pos,\
        cell_based=True,s_emiss=False,offset=0,parallel=False):
        time1 = time.time()
        self.time_0 = np.array([time.time()])
        dr,ray_ind = self.ray_trace_1(ll,ur,initial_pos,final_pos,\
                    cell_based=cell_based,parallel=parallel)
        time1 = time.time()-time1
        #print(rank,'ray_box',time1)
        sto_i={}
        if cell_based:
            dx = (ur[self.cells]-ll[self.cells]).mean(axis=1)
            #print(dx.shape,l_final_pos.shape)
        else:
            dx = 0
        sto_i['I'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['Q'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['U'] = np.zeros((len(l_final_pos),len(self.nu)))
        sto_i['V'] = np.zeros((len(l_final_pos),len(self.nu)))
        if parallel:
            sto_i['i'] = i_stars+offset
        if not parallel:
            sto_i['i'] = np.arange(len(l_final_pos))
        self.time_0 = np.append(self.time_0,time.time())
        tau_i_j,bigcount = self.ray_trace_4(final_pos,initial_pos,ray_ind,dr,densities,metals,star_vel)
        self.time_0 = np.append(self.time_0,time.time())
        sto_i = self.ray_trace_5(sto_i,final_pos,initial_pos,dx,dr,ray_ind,i_stars,star_vel,\
                tau_i_j,cell_based=cell_based,s_emiss=s_emiss)
        self.time_0 = np.append(self.time_0,time.time())
        self.time_0 = np.diff(self.time_0)
        print(rank,self.time_0/self.time_0.sum(),self.time_0.sum(),self.time_0.sum()/bigcount)
        # print(self.time_0,len(initial_pos),len(final_pos))
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
                                self.expand_spectra(self.spectra_i[j],i_stars[j])
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
        tau_i_j = cp.zeros((len(final_pos),len(initial_pos),len(self.nu)))
        red = self.redshift(initial_pos,final_pos,ray_ind,star_vel)
        #ind_all = np.unique(ray_ind[:,1])
        plot_files = np.unique(self.plotfile)
        bigcount = 0
        #time_piece = np.zeros(2)
        diff_nu = np.abs(np.diff(self.nu)/self.nu[1:]).min()
        bool_red = np.abs(red-1) > 0.5*diff_nu
        for i_file in plot_files:
            np.arange(len(metals))
            ind_true = np.arange(len(metals))[self.plotfile==i_file]
            ind_all_0 = ray_ind[:,1][np.isin(ray_ind[:,1],ind_true)]
            if len(ind_all_0 )>0:
                ind_all = np.unique(ind_all_0)
                # if i_file not in self.plot_t:
                #     self.plot_t[i_file] = np.load(i_file,allow_pickle=True).tolist()
                #time1 = time.time()
                #self.plot = np.load(i_file,allow_pickle=True).tolist()#self.plot_t[i_file]
                if self.gather:
                    self.plot = self.plot_hypes[i_file]
                else:
                    self.plot = np.load(i_file,allow_pickle=True).tolist()#self.plot_t[i_file]
                #time_piece[0] += time.time()-time1
                #time1 = time.time()
                self.temp_ind = np.unique(self.i_temp[ind_all])
                self.get_gas_rads(emiss=False)
                self.plot = None
                split_inds = np.array_split(ind_all,max(len(ind_all)/50,1))
                for ind_i,inds in enumerate(split_inds):
                    count = 0
                    chix = {}
                    temp_j = np.minimum(np.searchsorted(self.temp_ind,self.i_temp[inds]),len(self.temp_ind)-1)
                    # timei = time.time()
                    Z = metals[inds][:,np.newaxis]
                    DGRm = mH*10**(2.445*np.log10(Z)-2.029)
                    chix = self.chisdust_0*(self.temps[inds]<self.Tmax)[:,np.newaxis]*DGRm + self.chishe[temp_j] + 0.0204*Z*self.chismet_0[temp_j] +\
                                                self.chivdust_0*(self.temps[inds]<self.Tmax)[:,np.newaxis]*DGRm +\
                                                 self.chivhe[temp_j] + 0.0204*Z*self.chivmet_0[temp_j] +\
                                                    (self.elect_fract[inds]*6.652458e-25)[:,np.newaxis]
                    chix *= densities[inds,np.newaxis]/mH
                    #chix = {t: chix[i] for i,t in enumerate(inds)}
                    bool_in = np.isin(ray_ind[:,1],inds)
                    ray_ind_i = np.arange(len(ray_ind))[bool_in]
                    i_s, t_s, j_s = ray_ind[bool_in][:,0],ray_ind[bool_in][:,1],ray_ind[bool_in][:,2]
                    chi_ind = np.searchsorted(inds,t_s)
                    drt = dr[bool_in]
                    len_y = np.arange(len(i_s))
                    bool_y_red = np.logical_not(bool_red[ray_ind_i])
                    np.add.at(tau_i_j,(i_s[bool_y_red],j_s[bool_y_red]),drt[bool_y_red,np.newaxis]*(chix[chi_ind[bool_y_red]]))
                    for y in len_y[bool_red[ray_ind_i]]:
                        # if bool_red[ray_ind_i[y]]:
                            chix_t = cp.interp(self.nu,self.nu*red[ray_ind_i[y]],chix[chi_ind[y]])
                            tau_i_j[i_s[y],j_s[y]] += drt[y]*chix_t
                        # else:
                        #     tau_i_j[i_s[y],j_s[y]] += drt[y]*chix[chi_ind[y]]
                    count += bool_in.sum()
                    bigcount += count
                    chix = None
                self.deletechi()
                #time_piece[1] += time.time()-time1
            # timei = time.time()-timei
            # if ind_i%10 ==0 and bigcount >0:
            #     print(rank,ind_i,timei,len(split_inds),timei/count)
        chix = {}
        #print(time_piece/time_piece.sum())
        tau_i_j = np.exp(-tau_i_j)
        return tau_i_j,bigcount


    def ray_trace_5(self,sto_i,final_pos,initial_pos,dx,dr,ray_ind,i_stars,\
            star_vel,tau_i_j,cell_based=True,s_emiss=False):
        if cell_based:
            dist_arr = np.zeros((len(final_pos),len(initial_pos)))
            # for i in range(len(final_pos)):
            #     for j in range(len(initial_pos)):
            #         bool_i_j = (ray_ind[:,0] ==i)*(ray_ind[:,2] ==j)
            #         dist_arr[i][j] = dr[bool_i_j].sum()
            np.add.at(dist_arr,(ray_ind[:,0],ray_ind[:,2]),dr)
            mod = 1/(4*np.pi*dist_arr**2)
        if len(final_pos)>1:
            dist_arr = distance.cdist(final_pos,initial_pos)
        if not cell_based:
            dist_arr = np.linalg.norm(final_pos-initial_pos,axis=1)
        if not cell_based:
            mod = 1/(4*np.pi*dist_arr**2)
        if s_emiss:
            self.emission_2 = np.swapaxes(self.emission_2,0,1)
        dist_arr[np.isnan(dist_arr)] = 0
        for i in range(len(final_pos)):
            if len(final_pos)>1 or cell_based:
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
            red = (c_cgs + np.sign(v_dot)*np.linalg.norm(v_proj,axis=1))/c_cgs
            if s_emiss:
                Spectra_j = self.emission_2
            else:
                if self.new_stars:
                    self.expand_spectra(self.spectra_i,i_stars)
                    Spectra_j = self.Spectra
                else:
                    Spectra_j = self.Spectra[i_stars,:]
                self.Spectra = None
            if cell_based:
                bool_inside = (dist_arr[i,:] < dx[i])
                #print(dist_arr[i,:])
                modi[dist_arr[i,:]< 1] = 0
                if (dist_arr[i,:] < dx[i]).sum() >0:
                    modi[dist_arr[i,:] < dx[i]] = 4/(6*dx[i]**2)
                    tau_i_j[i,dist_arr[i,:] < dx[i]] = 1
            #print('1',Spectra_j)
            Spectra_j = np.array([[np.interp(self.nu,self.nu*red[y],Spectra_j[x][y]) \
                                     for y in range(len(Spectra_j[0]))] \
                                      for x in range(len(Spectra_j))])
            #print(np.sum(integrate.simpson(Spectra_j[:,:,self.bool_in]*modi[:,np.newaxis],self.nu[self.bool_in]),axis=1))
            mod_i_j = tau_i_j[i,:]*modi[:,np.newaxis]
            #print('2',Spectra_j)
            Spectra_j *= mod_i_j
            print(np.sum(integrate.simpson(Spectra_j[:,:,self.bool_in],self.nu[self.bool_in]),axis=1))
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
            bool_tmin = (tmin <= tmax)*(tmin <= 1)*(tmax>=0)
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
        red = (c_cgs + np.sign(v_dot)*np.linalg.norm(v_proj,axis=1))/c_cgs
        return red

    def find_line_ratio(self):
        current_file = self.plotfile[self.i_cen]
        prefix = f'{plotpath}/plothype_'
        suffix = '.npy'
        wav = 1e10*299792458./self.nu
        line_file = current_file[len(prefix):-len(suffix)]+'/linelist%s.txt' % self.i_temp[self.i_cen]
        line_list = np.loadtxt(self.absorb_path+line_file,dtype=str)
        indices = np.arange(len(line_list))
        h_index = indices[line_list[:,0]=='H'].min()
        met_index = indices[(line_list[:,0]=='C')*(line_list[:,2].astype(float)>1000)]
        lab = line_list[h_index][0]
        hwav = float(line_list[h_index][2])
        peak1 = float(line_list[h_index][3])
        wavenow1 = wav[np.argmin(np.abs(hwav-wav))]
        hint = self.Guasswav2(wavenow1,hwav,peak1,lab,self.temps[self.i_cen])
        metint = 0
        count = 0
        while metint < 1 and count <10:
            lab2 = line_list[met_index[count]][0]
            metwav = float(line_list[met_index[count]][2])
            peak2 = float(line_list[met_index[count]][3])
            wavenow2 = wav[np.argmin(np.abs(metwav-wav))]
            metint = self.Guasswav2(wavenow2,metwav,peak2,lab2,self.temps[self.i_cen])
            count += 1
        #print(count,self.i_temp[self.i_cen],lab,lab2,hwav,wavenow1,metwav,wavenow2,metint/hint,metint,hint)
        return np.argmin(np.abs(hwav-wav)),hint,np.argmin(np.abs(metwav-wav)),metint

    def Guasswav2(self,wavenow,center,peak,lab,temp):
        sol  = 299792458.
        center2 = sol/(center*1e-10)
        k = 1.38064852e-23
        amu = 1.66054e-27
        mass = pt.formula("%s" % lab).mass*amu
        sig = center2*np.sqrt(2*k*temp/(mass*sol**2))
        sig3 = center*sig/center2
        prefix = peak/(np.sqrt(np.pi)*sig3)
        expon = (.1/(sig3**2))
        f_out = prefix*np.exp(-expon*(wavenow-center)**2)
        #print(lab,(wavenow-center)/sig3,prefix)
        return f_out

    def set_emiss(self,densities,dx):
        self.mod_h1 = densities[self.i_cen]*dx[self.i_cen]**3
        self.mod_em = densities[self.i_cen]*dx[self.i_cen]**3
        tauhe = (self.chivhe)*self.rp*densities[self.i_cen]/mH
        taumet = (self.chivmet)*self.rp*densities[self.i_cen]/mH
        taudust = (self.chivdust)*self.rp*densities[self.i_cen]/mH
        if self.temps[self.i_cen]>=self.Tmax:
            self.emiss_dust = np.zeros(len(self.nu))
            nrg_dust = 0
        else:
            nrg_dust = self.I_f*(1-np.exp(-taudust))/self.mod_h1
            nrg_dust = integrate.simpson(nrg_dust,self.nu)
            dust_predicted = 4 * sigma * self.temps[self.i_cen]**4*self.planck_single(self.chivdust_0)
            #print(dust_predicted)
            dust_i = np.minimum(np.searchsorted(np.array(self.dust_nrg),dust_predicted),len(self.dust_nrg)-1)
            #print(dust_i,nrg_dust,self.dust_nrg,self.dust_nrg.shape)
            #print(dust_i,self.emiss_dust_0[dust_i].shape,self.dust_fix[dust_i].shape)
            emiss_dust_1 = np.interp(self.nu, self.dust_nu,self.emiss_dust_0[:,dust_i].flatten())
            self.emiss_dust = self.dust_fix[dust_i]*(nrg_dust/integrate.simpson(emiss_dust_1,\
                                    self.nu))*emiss_dust_1
        self.emisshe = self.emisshe_0#*self.blackbod(self.nu,self.temps[self.i_cen])
        self.emissmet = self.emissmet_0#*self.blackbod(self.nu,self.temps[self.i_cen])
        nrg_he = self.I_f*(1-np.exp(-tauhe))/self.mod_h1
        nrg_he = integrate.simpson(nrg_he,self.nu)
        self.emisshe = self.emisshe*nrg_he/integrate.simpson(self.emisshe,self.nu)
        ind_h,hint,ind_met,metint = self.find_line_ratio()
        rat_now = self.emissmet[ind_met]/self.emisshe[ind_h]
        rat_expected = (self.Z*metint)/hint
        self.emissmet = self.emissmet*(rat_expected/rat_now)/integrate.simpson(self.emissmet,self.nu)
        #print(rat_expected/rat_now)
        #nrg_met = self.I_f*(1-np.exp(-taumet))/self.mod_h1
        #nrg_met = integrate.simpson(nrg_met,self.nu)
        cmbhe = integrate.simpson(self.cmb*(1-np.exp(-tauhe)),self.nu)
        cmbmet = integrate.simpson(self.cmb*(1-np.exp(-taumet)),self.nu)
        cmbdust = integrate.simpson(self.cmb*(1-np.exp(-taudust)),self.nu)
        sumcmb = integrate.simpson(self.cmb,self.nu)
        self.cmb_abs = np.array([cmbhe/sumcmb,cmbmet/sumcmb,cmbdust/sumcmb])
        # I_e = integrate.simpson(self.I_f,self.nu)
        # emitting = integrate.simpson(self.I_f*(1-np.exp(-tauhe-taumet-taudust)),self.nu)
        # print(self.i_cen,self.cen_index,(self.rp).mean(),nrg_he*self.mod_h1/1e33,nrg_met*self.mod_h1/1e33,nrg_dust*self.mod_h1/1e33,I_e/1e33,emitting/1e33)
        # self.emissmet = self.emissmet*nrg_met
        self.emiss = self.emisshe+self.emissmet
        # print(self.emiss)
        # self.emiss *= len(temps)/(len(self.emission))
        # print(self.emiss)

        #print(self.emiss)

    def set_absorb(self):
        Z_sun = self.Z
        DGR = 10**(2.445*np.log10(self.Z)-2.029)
        self.chisdust = (self.chisdust_0*(self.temps[self.i_cen]<self.Tmax) * DGR * mH).flatten()
        self.chivdust = (self.chivdust_0*(self.temps[self.i_cen]<self.Tmax) * DGR * mH).flatten()
        self.chivmet = self.chivmet_0 * Z_sun * 0.0204
        self.chismet = self.chismet_0 * Z_sun * 0.0204
        self.chis = self.chishe + self.chismet + self.chisdust
        self.chiv = self.chivhe + self.chivmet + self.chivdust

    def deletechi(self):
        self.chivhe = None
        self.chishe = None
        self.chivmet_0 = None
        self.chismet_0 = None

    def get_gas_rads_all(self):
        self.plot2 = np.zeros((6,len(self.nu)))
        #print(self.plot2[0].shape,self.plot['chivhe'][...,self.bool_nu][self.temp_ind].shape)
        self.plot2[0] = self.plot['chivhe'][...,self.bool_nu][self.temp_ind]
        self.plot2[1] = self.plot['chishe'][...,self.bool_nu][self.temp_ind]
        self.plot2[1] -= self.plot2[1].min()
        thomson = self.elect_fract[self.i_cen]*6.652458e-25
        self.plot2[1] += thomson
        self.plot2[2] = self.plot['chivmet'][...,self.bool_nu][self.temp_ind]
        self.plot2[3] = self.plot['chismet'][...,self.bool_nu][self.temp_ind]
        self.plot2[3] -= self.plot2[3].min()
        self.plot2[4] = self.plot['emisshe'][...,self.bool_nu][self.temp_ind]/self.nu
        self.plot2[5] = self.plot['emissmet'][...,self.bool_nu][self.temp_ind]/self.nu

    def get_gas_rads_all_2(self):
        self.chivhe = self.plot2[0]
        self.chishe = self.plot2[1]
        self.chivmet_0 = self.plot2[2]
        self.chismet_0 = self.plot2[3]
        self.emisshe_0 = self.plot2[4]
        self.emissmet_0 = self.plot2[5]


    def get_gas_rads(self,emiss=True):
        self.chivhe = self.plot['chivhe'][...,self.bool_nu][self.temp_ind,...]
        self.chishe = self.plot['chishe'][...,self.bool_nu][self.temp_ind,...]
        self.chishe -= self.chishe.min()
        self.chivmet_0 = self.plot['chivmet'][...,self.bool_nu][self.temp_ind,...]
        self.chismet_0 = self.plot['chismet'][...,self.bool_nu][self.temp_ind,...]
        self.chismet_0 -= self.chismet_0.min()
        if emiss:
            self.emisshe_0 = self.plot['emisshe'][...,self.bool_nu][self.temp_ind,...]/self.nu
            self.emissmet_0 = self.plot['emissmet'][...,self.bool_nu][self.temp_ind,...]/self.nu

    def clean_vars(self):
        self.chisdust, self.chivdust, self.chivmet, self.chismet, self.chis, self.chiv =\
            None, None, None, None, None, None
        self.Scattered, self.Stokes_f, self.Emitted, self.Atten, self.Initial = None, None, None, None, None
        self.lams, self.dust_lams, self.Ns_all_dust, self.f_em, self.rp, \
            self.redistem, self.redistI, self.redistQ, self.redistU, self.redistV =\
                None, None, None, None, None, None, None, None, None, None
        self.emiss, self.emisshe, self.emissmet, self.I_f, self.Q_f, \
            self.U_f, self.V_f = None, None, None, None, None, None, None
        self.Ns_all,self.thomson = None,None
        self.chivmet_0, self.chismet_0, self.emisshe_0, self.emissmet_0 = None, None, None, None
        self.dust_count = None


    def blackbod(self,nu,temp):
          h = 6.62607015e-34
          c = 299792458
          k = 1.380649e-23
          a = np.ones(temp.shape)
          nu2 = a[...,np.newaxis]*nu
          pre = (2*h*nu2**3)/c**2
          den = np.exp(h*nu2/(k*temp[...,np.newaxis])) - 1
          black = pre/den
          black *=  1000
          return black

    def planck(self,kappa):
           b_nu = self.blackbod(self.nu,self.temp)
           planck1 = integrate.simpson(b_nu*kappa,self.nu)/integrate.simpson(b_nu,self.nu)
           return planck1

    def blackbod_single(self):
          h = 6.62607015e-34
          c = 299792458
          k = 1.380649e-23
          pre = (2*h*self.nu**3)/c**2
          den = np.exp(h*self.nu/(k*self.temps[self.i_cen])) - 1
          black = pre/den
          black *=  1000
          return black

    def planck_single(self,kappa):
           b_nu = self.blackbod_single()
           planck1 = integrate.simpson(b_nu*kappa,self.nu)/integrate.simpson(b_nu,self.nu)
           return planck1

    def get_spos(self):
        self.spos,self.svel,self.bool_inside,\
            self.halo_c,self.halo_v,self.halo_r,\
            self.plot,self.nu,self.temp,self.lums = None, None, None, None ,None, None,None, None, None, None
        self.bool_nu = None
        if rank ==0:
            stars = np.load(self.star_folder+'starlists_2013.npy',allow_pickle=True).tolist()
            self.spos = stars[self.halo][self.timestep]['positions2']*self.ds.length_unit.in_units('cm').v
            self.svel = stars[self.halo][self.timestep]['vels2']
            print('pop2',len(self.spos),'Stars')
            #self.lums = stars[self.halo][self.timestep]['luminosity2']
            self.spos = np.concatenate((self.spos,stars[self.halo][self.timestep]['positions3']*self.ds.length_unit.in_units('cm').v),axis=0)
            self.svel = np.concatenate((self.svel,stars[self.halo][self.timestep]['vels3']),axis=0)
            #self.lums = np.append(self.lums,stars[self.halo][self.timestep]['luminosity3'])
            print('pop3',len(self.spos),'Stars')
            stars = 0
            halotree = np.load(self.star_folder+'halotree_2013_final.npy',allow_pickle=True).tolist()
            self.halo_c = halotree[self.halo][self.timestep]['Halo_Center']*self.ds.length_unit.in_units('cm').v #np.average(self.spos,axis=0,weights=self.lums)#
            self.halo_v = halotree[self.halo][self.timestep]['Vel_Com']*self.ds.length_unit.in_units('cm').v
            self.halo_r = halotree[self.halo][self.timestep]['Halo_Radius']*self.ds.length_unit.in_units('cm').v #1.5*self.spos.std(axis=0).mean()#
            self.bool_inside = (np.sum(self.spos > self.halo_c-self.halo_r,axis=1)==3)*\
                        (np.sum(self.spos < self.halo_c+self.halo_r,axis=1)==3)
            print(self.halo_c,self.halo_r,self.spos)
            self.spos = self.spos[self.bool_inside]
            print('inside',len(self.spos),'Stars')
            self.svel = self.svel[self.bool_inside]
            #self.lums = self.lums[self.bool_inside]
            print(len(self.spos),"Stars")
            halotree = None
            self.plot = np.load('%s/plothype_0_0_0.npy' % plotpath,allow_pickle=True).tolist()
            self.nu = self.plot['nu']
            wav = 2.998e+14/self.nu
            #print(wav)
            self.bool_nu = np.arange(len(wav))[wav <1e3]
            self.nu = self.nu[self.bool_nu]
            self.temp = self.plot['temp']
        self.spos,self.svel,self.halo_c,self.halo_v,self.halo_r,self.plot,\
                self.nu,self.temp,self.bool_inside,self.lums,self.bool_nu =comm.bcast((self.spos,self.svel,\
                    self.halo_c,self.halo_v,self.halo_r,self.plot,self.nu,self.temp,self.bool_inside,self.lums,self.bool_nu),root=0)

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
        file = np.array([f'{plotpath}/plothype{id_cen[idx][0]}_{id_cen[idx][1]}.npy'])
        return file

    def find_all_plothype_file(self,I):
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

        wav = c/self.nu
        boolion = wav <911
        lumion = integrate.simpson(I[:,boolion],self.nu[boolion])
        files = np.array([])
        centers = np.load(f'{plotpath}/centers.npy',allow_pickle=True).tolist()
        id_cen = np.load(f'{plotpath}/id_cen.npy',allow_pickle=True).tolist()
        for i in range(len(I)):
            ph_i = closest(wav, 900, I[i])
            ph_uv = closest(wav, 1500, I[i])
            ph_b1 = closest(wav, 1300, I[i]*self.nu/wav)
            ph_b2 = closest(wav, 3500, I[i]*self.nu/wav)
            fluxes = np.array([ph_i,ph_uv,ph_b1,ph_b2,lumion[i]])
            Beta,psi_ion = self.find_slope(fluxes[:,np.newaxis])[0]
            dist = distance.cdist(np.array([Beta,psi_ion])[:,np.newaxis].T,np.array(centers))
            idx = np.argmin(dist,axis=1)[0]
            files = np.append(files,np.array([f'{plotpath}/plothype{id_cen[idx][0]}_{id_cen[idx][1]}.npy']))
            #print(Beta,psi_ion,f'{plotpath}/plothype{id_cen[idx][0]}_{id_cen[idx][1]}.npy')
        files = np.array(files,dtype=object)
        return files

    def find_fluxes_lion(self,I,freq):
        h = 6.626e-27
        c = 2.998e+18
        wav = c/freq
        boolion = wav <911
        lumion = np.abs(integrate.simpson(I[:,boolion],freq[boolion]))
        n_phot = np.abs(integrate.simpson(I[:,boolion]/(h*freq[boolion]),freq[boolion]))
        lumtot = np.abs(integrate.simpson(I,freq))
        fluxes = []
        for i in range(len(I)):
            ph_i = closest(wav, 900, I[i])
            ph_uv = closest(wav, 1500, I[i])
            ph_b1 = closest(wav, 1300, I[i]*freq/wav)
            ph_b2 = closest(wav, 3500, I[i]*freq/wav)
            fluxes.append([ph_i,ph_uv,ph_b1,ph_b2,lumion[i],lumtot[i],n_phot[i]])
        fluxes = np.array(fluxes)
        return fluxes


    def find_initial_files(self,h1den,densities,dx,ll,ur):
        fluxes = None
        if rank ==0:
            time_8 = time.time()
            fluxes = self.find_fluxes_lion(self.spectra,self.freq)
        fluxes = comm.bcast(fluxes,root=0)
        initial_pos = self.spos
        self.find_all_lums()
        split_inds_1 = np.array_split(np.arange(len(ll)),max(len(ll)/50,nprocs))
        jobs,sto = job_scheduler_2(np.arange(len(split_inds_1)))
        ranks = np.arange(nprocs)
        # self.pos_grid,self.Grid = self.build_hierarchy(initial_pos)
        dummy = 0
        for ranki in np.arange(nprocs):
            dummy = comm.bcast(dummy,root=ranki)
        job_i = 0
        Done = np.full(nprocs,False)
        while not Done[rank]:
            rank_now,job_i,Done = job_organizer2(job_i,Done,len(sto))
            if rank == rank_now:
                    if (job_i+1)%10==0:
                        print('Finding %s out of %s' % (job_i+1,len(split_inds_1)))
                    time_list = np.array([time.time()])
                    final_pos = (ur[split_inds_1[job_i]]+ll[split_inds_1[job_i]])/2
                    dr,ray_ind = self.ray_trace_1(ll,ur,initial_pos,final_pos,\
                                cell_based=True,parallel=False)
                    time_list = np.append(time_list,time.time())
                    tau_i_j = cp.zeros((len(final_pos),len(initial_pos)))
                    ind_all = np.unique(ray_ind[:,1])
                    split_inds = np.array_split(ind_all,max(len(ind_all)/500,1))
                    dist_arr = np.zeros((len(final_pos),len(initial_pos)))
                    np.add.at(dist_arr,(ray_ind[:,0],ray_ind[:,2]),dr)
                    for ind_i,inds in enumerate(split_inds):
                        bool_in = np.isin(ray_ind[:,1],inds)
                        i_s, t_s, j_s = ray_ind[bool_in][:,0],ray_ind[bool_in][:,1],ray_ind[bool_in][:,2]
                        drt = dr[bool_in]
                        len_y = np.arange(len(i_s))
                        chix_t = h1den[inds]*1.78e-18/mH
                        chi_ind = np.searchsorted(inds,t_s)
                        np.add.at(tau_i_j,(i_s,j_s),drt*chix_t[chi_ind])
                    tau_i_j = np.exp(-tau_i_j)
                    mod = 1/(4*np.pi*dist_arr**2)
                    dist_arr = distance.cdist(final_pos,initial_pos)
                    final_fluxes = np.zeros((len(final_pos),7))
                    time_list = np.append(time_list,time.time())
                    for i in range(len(final_pos)):
                        modi = mod[i,:]
                        tau_i_j[i,dist_arr[i,:] < dx[i]] = 1
                        mod_i_j = tau_i_j[i,:]*modi
                        final_fluxes[i,:] = np.sum(fluxes*modi[:,np.newaxis],axis=0)
                        final_fluxes[i,-3] = np.sum(fluxes[:,-3]*mod_i_j)
                        final_fluxes[i,-1] = np.sum(fluxes[:,-1]*mod_i_j)
                    final_fluxes = np.maximum(1e-100,final_fluxes)
                    Beta,psi_ion = self.find_slope(final_fluxes[:,:-2])
                    #time_list = np.append(time_list,time.time())
                    cm_mod = (3/2)/c_cgs
                    U = final_fluxes[i,-1]*cm_mod/(densities[split_inds_1[job_i]]/mH)
                    U = np.maximum(1e-20,U)
                    sto[job_i]['filenames'] = self.find_new_plothype(Beta,psi_ion,np.log10(U))
                    time_list = np.append(time_list,time.time())
                    if (job_i+1)%10==0:
                        print(job_i+1,np.diff(time_list))
                    jobs[rank_now].append(job_i)
        for rank_i in jobs:
            jobs[rank_i] = comm.bcast(jobs[rank_i],root=rank_i)
        filenames = np.array([], dtype='U%s' % (len(plotpath)+len('/plothype_00_00_00.npy')))
        for ranki in np.arange(nprocs):
            for j in jobs[ranki]:
                sto[j] = comm.bcast(sto[j],root=ranki)
        for j in range(len(split_inds_1)):
            filenames = np.append(filenames,sto[j]['filenames'])
        if rank ==0:
            print('Preconditioning Done, Time: %s' % (time.time()-time_8))
        return filenames

    def find_new_files(self,I,densities,freq):
        fluxes = self.find_fluxes_lion(I,freq)
        cm_mod = (3/2)/c_cgs
        if fluxes.ndim ==2:
            Beta,psi_ion = self.find_slope(fluxes[:,:-2])
            U = fluxes[:,-1]*cm_mod/(densities/mH)
        else:
            Beta,psi_ion = self.find_slope(fluxes[:-2])
            U = fluxes[-1]*cm_mod/(densities/mH)
            Beta,psi_ion,U = np.array([Beta]),np.array([psi_ion]),np.array([U])
        U = np.maximum(1e-20,U)
        filenames = self.find_new_plothype(Beta,psi_ion,np.log10(U))
        return filenames

    def find_new_plothype(self,Beta,psi_ion,U):
        clusters = np.load(f'{plotpath}/clusters.npy',allow_pickle=True).tolist()
        Index = np.zeros((len(Beta),3))
        Index[:,0] = 2
        filenames = np.array([], dtype='U%s' % (len(plotpath)+len('/plothype_00_00_00.npy')))
        for ind in clusters:
            bool_ind = (Index[:,0] >= ind)*(U >= clusters[ind]['Urange'][0])
            Index[bool_ind,0] = ind
            true_means = clusters[ind]['true_means']
            true_means[:,0] *= 2
            dist = distance.cdist(np.array([2*Beta[bool_ind],psi_ion[bool_ind]]).T,true_means)
            idx = np.argmin(dist,axis=1)
            Index[bool_ind,1] = idx
            ind_range = np.arange(len(Index))
            for idx_i in np.unique(idx):
                bool_idx = idx == idx_i
                dist_2 = np.linalg.norm(U[bool_ind][bool_idx][:,np.newaxis]-clusters[ind]['logU'][idx_i],axis=0)
                idx_2 = np.argmin(dist_2)
                Index[ind_range[bool_ind][bool_idx],2] = idx_2
        #print(Index)
        Index = Index.astype(int)
        for i in range(len(Index)):
            filenames = np.append(filenames,np.array([f'{plotpath}/plothype_{Index[i][0]}_{Index[i][1]}_{Index[i][2]}.npy']))
        #print(np.unique(filenames))
        return filenames


    def find_slope(self,fluxes):
        Beta = np.log(fluxes[:,3]/\
            fluxes[:,2])/np.log(slope)
        psi_ion = np.log10(fluxes[:,4]/\
            (2e15*fluxes[:,1]))
        return Beta,psi_ion


    def get_stars(self,subset=[]):
        self.spectra = ([])
        ages = self.stars['age2']
        creation_times = self.ds.current_time.in_units('Gyr').v - ages
        metallicities = self.stars['met2']
        masses = self.stars['mass2']
        masses = masses[self.bool_inside[np.arange(len(masses))]]
        ages = ages[self.bool_inside[np.arange(len(masses))]]
        creation_times = creation_times[self.bool_inside[np.arange(len(masses))]]
        metallicities = metallicities[self.bool_inside[np.arange(len(masses))]]
        if len(subset) != 0:
            masses = masses[subset]
            ages = ages[subset]
            creation_times = creation_times[subset]
            metallicities = metallicities[subset]
        lums, self.freq, spectra_i =\
            SSP_interpolator(self.path_to_fsps,self.ds,self.path_to_fsps,ages,creation_times,metallicities,masses)
        self.spectra = spectra_i * 3.828e33
        if len(self.stars['age3']) >0:
            self.popIIIsum()

    def popIIIsum(self,cutoff=55):
            rad = np.load(self.path_to_fsps+'rad_array.npy',allow_pickle=True).tolist()
            age = self.stars['age3']
            mass = self.stars['mass3']
            bool1 = (mass >cutoff)
            bool2 =(mass <= cutoff)
            lenage = rad['PopIII.1'].shape[0]
            ind1 = np.minimum(np.searchsorted(rad['PopIIIages'],age),lenage-1)
            ind1a = ind1[bool1]
            ind1b = ind1[bool2]
            nrg1 = rad['PopIII.1'][ind1a]*mass[bool1,np.newaxis]
            nrg2 = rad['PopIII.2'][ind1b]*mass[bool2,np.newaxis]
            if nrg1.shape[0] == 0:
                nrgt = nrg2
            elif nrg2.shape[0] == 0:
                nrgt = nrg1
            else:
                nrgt = np.concatenate((nrg1,nrg2),axis=0)
            spectra = np.array([np.interp(self.freq, rad['nu'], nrgt[i]) for i in range(len(nrgt))])* 3.828e33
            if len(self.spectra) ==0:
                self.spectra = spectra
            else:
                self.spectra = np.concatenate((self.spectra,spectra),axis=0)

    def expand_spectra(self,spectra,i_star):
        spectra  =  np.array([np.interp(self.nu, self.freq[::-1], spectra[i][::-1]) for i in range(len(spectra))])
        I = spectra
        Q = spectra * self.randQ_star[i_star][:,np.newaxis]
        U = spectra * self.randU_star[i_star][:,np.newaxis]
        V = np.zeros(spectra.shape)
        self.Spectra = np.zeros((4,len(i_star),len(self.nu)))
        #print(I,I.shape,self.Spectra,i_star)
        #print(i_star,self.randQ_star[i_star],self.randU_star[i_star])
        self.Spectra[0] = I
        self.Spectra[1] = Q
        self.Spectra[2] = U
        self.Spectra[3] = V

    def get_grid_values(self):
        lu = self.ds.length_unit.in_units('cm').v
        reg = self.ds.region(self.halo_c/lu,(self.halo_c-self.halo_r)/lu,\
                        (self.halo_c+self.halo_r)/lu)
        # if rank ==0:
        #     print(self.ds.derived_field_list)
        densities = (reg['HI_Density']+reg['HII_Density']+2*reg['H2I_Density']+2*reg['H2II_Density']).in_units('g/cm**3')
        dx = reg['dx'].in_units('cm')
        temps = reg['temperature'].in_units('K')
        metals = reg['metallicity'].in_units('Zsun')
        h1den = reg['HI_Density'].in_units('g/cm**3')
        elect_fract = reg['El_number_density'].in_units('1/cm**3').v/(densities.v/mH)
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
        return densities.v,dx.v,temps.v,metals.v,ll.v,ur.v,v.v,h1den.v,elect_fract


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
        Stokes_f = np.zeros((4,len(self.P2)))
        if self.Ns_all_dust.max() >0:
            Stokes_i = np.zeros((4,len(self.P2)))
            # taud = (self.chisdust)*self.rp*densities[self.i_cen]/mH
            # taud = np.maximum(taud,1e-10)
            # absorb = 1-np.exp(-taud)
            absorb = 1
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
            for i in range(min(int(self.Ns_all_dust.max()+1),100)):
                bool_Ns = self.Ns_all_dust>i
                tauv = (self.chiv)*np.minimum(self.dust_lams,\
                    dx[self.i_cen])*densities[self.i_cen]/mH
                absorb = np.exp(-tauv)[bool_Ns]
                Stokes_f[:,bool_Ns] = \
                    (s_0.T * Stokes_i).sum(axis=1)[:,bool_Ns]*(np.minimum(self.Ns_all_dust[bool_Ns]-i,1))*absorb \
                    + Stokes_f[:,bool_Ns]*(1-np.minimum(self.Ns_all_dust[bool_Ns]-i,1))
                #print(i,'dust',self.Ns_all_dust.max()-i,self.temps[self.i_cen])
                s_0 = np.matmul(Scatter,s_0)
        self.Stokes_f = Stokes_f

    def find_thomson(self,densities,dx):
        Scatter = np.zeros((4,4))
        Stokes_i = np.zeros((4,len(self.nu)))
        # taud = (self.chisdust)*self.rp*densities[self.i_cen]/mH
        # taud = np.maximum(taud,1e-10)
        # absorb = 1-np.exp(-taud)
        if self.temps[self.i_cen] >1e6:
            me = 9.1093837e-31
            cs = 299792458
            kb = (3/2)*1.380649e-23
            num = me*cs**2
            mu = num/(self.temps[self.i_cen]*kb)
            Gn = kn(1,mu)/(mu*kn(2,mu))
            if np.isnan(Gn):
                Gn = 0
        else:
            Gn = 0
        # print(Gn,mu)
        absorb = 1
        Stokes_i[0] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb + (self.I_f+self.redistI)*absorb \
                + self.mod_em*(1-self.f_em)*(self.emiss+self.redistem)*absorb
        Stokes_i[1] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb*self.randQ[self.i_cen] + \
                    (self.Q_f+self.redistQ)*absorb
        Stokes_i[2] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb*self.randU[self.i_cen] + (self.U_f+self.redistU)*absorb
        Stokes_i[3] = (1-self.f_em)*self.mod_h1*self.emiss_dust*0*absorb + (self.V_f+self.redistV)*absorb
        Scatter[0,0] = 5/6 + Gn*1/3
        Scatter[1,1] = 5/6 + Gn*1/3
        Scatter[0,1] = 1/6
        Scatter[1,0] = 1/6
        Scatter[2,2] = Gn*2/3
        s_0 = Scatter
        Stokes_f = np.zeros((4,len(self.nu)))
        for i in range(min(int(self.Net+1),100)):
            tauv = (self.chiv)*np.minimum(self.dust_lams,\
                dx[self.i_cen])*densities[self.i_cen]/mH
            absorb = np.exp(-tauv)
            Stokes_f = \
                (s_0.T[...,np.newaxis] * Stokes_i).sum(axis=1)*(np.minimum(self.Net-i,1))*absorb \
                + Stokes_f*(1-np.minimum(self.Net-i,1))
            #print(i,'thomson',self.Net-i,self.temps[self.i_cen])
            s_0 = np.matmul(Scatter,s_0)
        self.Stokes_t = Stokes_f
        self.Net, self.t_lams = None, None


    def find_Scattered(self,densities,dx):
        Scattered = np.zeros((4,len(self.P2)))
        tau = (self.chiv)*self.rp*densities[self.i_cen]/mH
        tau = np.maximum(tau,1e-10)
        taud = (self.chisdust+self.thomson)*self.rp*densities[self.i_cen]/mH
        taud = np.maximum(tau,1e-10)
        absorb = np.exp(-taud)*np.exp(-tau)
        #pass_through = np.exp(-tau)
        Scattered[0] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb +\
                self.mod_em*(1-self.f_em)*(self.emiss+self.redistem)*absorb +\
                (1-self.f_em)*(self.redistI)*absorb + self.I_f*self.Ns_all*absorb
        Scattered[1] = (1-self.f_em)*self.mod_h1*self.emiss_dust*self.randQ[self.i_cen]*absorb +\
               (1-self.f_em)*(self.redistQ)*absorb + self.Q_f*self.Ns_all*absorb
        Scattered[2] = (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb*self.randU[self.i_cen] +\
               (1-self.f_em)*(self.redistU)*absorb + self.U_f*self.Ns_all*absorb
        Scattered[3] = (1-self.f_em)*self.mod_h1*self.emiss_dust*0*absorb +\
               (1-self.f_em)*(self.redistV)*absorb + self.V_f*self.Ns_all*absorb
        self.Scattered = Scattered
        self.dust_count += (1-self.f_em)*self.mod_h1*self.emiss_dust*absorb


    def find_Emitted(self,dx,densities):
        Emitted = np.zeros((4,len(self.P2)))
        d_em = np.minimum(self.lams,dx[self.i_cen])/2
        tau = (self.chiv)*d_em*densities[self.i_cen]/mH
        tau = np.maximum(tau,1e-10)
        absorb = (1-np.exp(-tau))/tau
        Emitted[0] = self.f_em*self.mod_h1*self.emiss_dust*absorb +\
                self.mod_em*(self.f_em)*(self.emiss+self.redistem)*absorb + \
                    (self.f_em)*(self.redistI)*absorb
        Emitted[1] = self.f_em*self.mod_h1*self.emiss_dust*absorb*self.randQ[self.i_cen] + \
                    (self.f_em)*(self.redistQ)*absorb
        Emitted[2] = self.f_em*self.mod_h1*self.emiss_dust*absorb*self.randU[self.i_cen] + \
                    (self.f_em)*(self.redistU)*absorb
        Emitted[3] = self.f_em*self.mod_h1*self.emiss_dust*0*absorb + \
                    (self.f_em)*(self.redistV)*absorb
        self.Emitted = Emitted
        self.dust_count = self.f_em*self.mod_h1*self.emiss_dust*absorb
        #print(self.mod_em,self.emiss.sum(),absorb.sum(),tau)


    def find_Atten(self,densities,dx):
        self.Atten = np.zeros((4,len(self.P2)))
        d_em = np.minimum(self.lams,dx[self.i_cen])/2
        tau = (self.chiv+self.chis)*dx[self.i_cen]*densities[self.i_cen]/mH
        tau = np.maximum(tau,1e-10)
        absorb = np.exp(-tau)
        self.Atten[0] = self.I_f*absorb
        self.Atten[1] = self.Q_f*absorb
        self.Atten[2] = self.U_f*absorb
        self.Atten[3] = self.V_f*absorb
        self.Initial = np.zeros((4,len(self.P2)))
        self.Initial[0] = self.I_f
        self.Initial[1] = self.Q_f
        self.Initial[2] = self.U_f
        self.Initial[3] = self.V_f

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
            plt.ylim(self.Stokes_f.max()*1e-10,max(1e15,self.Stokes_f.max())*5)
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



        h = 1
        for i in range(len(self.Atten)):
            if self.Atten[i].max()>0:
                plt.plot(wav,np.abs(self.Atten[i])/(h*dx[self.i_cen]**3),label='%s Attenuated' %labs[i],color=colors[i])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Radiant Density [erg/s/cm$^3$/Hz]')
        plt.legend(fontsize='x-small')
        plt.xlim(1e-2,1e3)
        plt.ylim(1e-50,self.Atten.max()*5/(dx[self.i_cen]**3))
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
        plt.ylim(1e4,self.Scattered.max()*5)
        plt.savefig(self.plot_path+'Scattered_Lines_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()



        tot_emission = (self.Stokes_f+self.Emitted+self.Atten+self.Scattered)/(dx[self.i_cen]**3)
        for i in range(len(tot_emission)):
            if self.Initial[i].max()>0:
                plt.plot(wav,np.abs(self.Initial[i])/(h*dx[self.i_cen]**3),':',label='%s External' %labs[i],color=colors[i],linewidth=0.3)
            if tot_emission[i].max()>0:
                plt.plot(wav,np.abs(tot_emission[i]),label='%s Processed' %labs[i],color=colors[i],linewidth=0.9)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Radiant Density [erg/s/cm$^3$/Hz]')
        plt.legend(fontsize='x-small')
        plt.ylim(1e-45,tot_emission.max()*5)
        plt.xlim(1e-2,1e3)
        plt.savefig(self.plot_path+'Total_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()


        tot_emission = (self.Stokes_f+self.Emitted+self.Atten+self.Scattered+self.Stokes_t)/(h*dx[self.i_cen]**3)
        for i in range(len(tot_emission)):
            if self.Initial[i].max()>0:
                plt.plot(wav,np.abs(self.Initial[i])/(h*dx[self.i_cen]**3),':',label='%s External' %labs[i],color=colors[i],linewidth=0.3)
            if tot_emission[i].max()>0:
                plt.plot(wav,np.abs(tot_emission[i]),label='%s Processed' %labs[i],color=colors[i],linewidth=0.9)
        #plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wavelength [micron]')
        plt.ylabel(r'Radiant Density [erg/s/cm$^3$/Hz]')
        plt.legend(fontsize='x-small')
        plt.ylim(1e-43,tot_emission[:,wav<1].max()*5)
        plt.xlim(1e-1,1)
        plt.savefig(self.plot_path+'TotalUV_%s_%s.pdf' % (self.i_cen,self.plot_ind))
        plt.clf()




        tot_emission = self.Stokes_f+self.Emitted+self.Scattered+self.Stokes_t
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


def job_organizer(root_ranks,job_i,Done,len_jobs,or_root=1):
    root_now = -1
    rank_now = -1
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
        if Done.sum() == len(Done)-1:
            Done[or_root] = True
        else:
            req = comm.irecv(tag=13,source=MPI.ANY_SOURCE)
            root_now = req.wait()
        if Done[bool_active].sum() == len(Done[bool_active]):
            Done[root_now] = True
        else:
            req = comm.irecv(tag=12,source=MPI.ANY_SOURCE)
            rank_now = req.wait()
        if job_i>=len_jobs:
            Done[rank_now] = True
            rank_now_i = -1
            root_now_i = -1
        else:
            rank_now_i = rank_now
            root_now_i = root_now
        if not Done[or_root]:
            req = comm.Send((Done),tag=20,dest=root_now)
        if Done[root_ranks].sum() ==0:
            req = comm.Send((Done),tag=21,dest=rank_now)
        if not Done[or_root]:
            if Done[bool_active].sum() != len(Done[bool_active]) and not Done[rank_now]:
                req = comm.isend(rank_now_i,tag=15,dest=rank_now)
                req.wait()
                req = comm.isend(job_i,tag=16,dest=rank_now)
                req.wait()
                req = comm.isend(root_now,tag=19,dest=rank_now)
                req.wait()
            if not Done[root_now]:
                req = comm.isend(job_i,tag=17,dest=root_now)
                req.wait()
                req = comm.isend(rank_now,tag=18,dest=root_now)
                req.wait()
                req = comm.isend(root_now_i,tag=14,dest=root_now)
                req.wait()
        #print(rank_now,root_now,job_i,Done)
        job_i += 1
    return rank_now,root_now,job_i,Done,time3

def job_organizer2(job_i,Done,len_jobs,or_root=0):
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
        bool_active = np.logical_not(np.isin(np.arange(nprocs),or_root))
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
            if Done[bool_active].sum() != len(Done[bool_active]) and not Done[rank_now]:
                req = comm.isend(rank_now_i,tag=15,dest=rank_now)
                req.wait()
                req = comm.isend(job_i,tag=16,dest=rank_now)
                req.wait()
        #print(rank_now,root_now,job_i,Done)
        job_i += 1
    return rank_now,job_i,Done

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
            # np.random.shuffle(out_list_2)
            for o in ranks:
                if count + out_list_2[o] < len(out_list):
                    i = count + out_list_2[o].item()
                    jobs[o].append(out_list[i])
            count += len(ranks)
        # for o in jobs:
        #     np.random.shuffle(jobs[o])
    jobs = comm.bcast(jobs, root=0)
    return jobs, sto

def closest(lst, K, olist):
    idx = (np.abs(lst - K)).argmin()
    return olist[idx]

def job_scheduler_2(out_list,ranklim=1e99):
    ranks = np.arange(min(nprocs,ranklim)).astype(int)
    #print(ranks)
    jobs = {i.item(): [] for i in ranks}
    sto = {t: {} for t in out_list}
    return jobs, sto

def make_sample():
    stars = np.load(ds_path_0+savestring+'/starlists_%s.npy' % halo_version,allow_pickle=True).tolist()
    sample = {}
    for halo in stars:
        times = np.array(list(stars[halo].keys()))
        sample[halo] = times[::10]
    np.save(savestring+'/sample.npy',sample)
    return sample

if __name__ == "__main__":
        halo_version = 2013
        test_num = sys.argv[1]
        savestring = sys.argv[2]
        halonum  = sys.argv[3]
        delta = False
        cuda = False
        if cuda:
            import cupy as cp
        else:
            import numpy as cp
        if delta:
            plotpath = '/work/hdd/bezm/gtg115x/Analysis/Rad_Trans_2025'
            ds_path_0 = '/work/hdd/bezm/gtg115x/TreesandLists/'
        else:
            plotpath = 'Cloudy'
            ds_path_0 = '/Users/kirkbarrow/Research_Mentorship/a_Edward/simfiles/'
        if not os.path.exists(savestring+'/sample.npy'):
            sample = {}
            if rank ==0:
                sample = make_sample()
            sample = comm.bcast(sample,root=0)
        else:
            sample = np.load(savestring+'/sample.npy',allow_pickle=True).tolist()
        if rank ==0:
            print(halonum)
            print(list(sample.keys()))
        c = 2.998e+18
        mH = 1.67e-24
        pc = 3.086e+18
        sigma = 5.67051e-5
        c_cgs =  2.99792458e10
        slope = 3500/1500
        cmb = True
        redo_fig = True
        eps = np.finfo(np.float32).eps
        for timenum in sample[halonum]:
            if rank==0:
                print(timenum)
            Radiative_Transfer(halonum,int(timenum))
        #Radiative_Transfer('7',5)
        # if delta:
        #Radiative_Transfer('4',422)
        # else:
        #    Radiative_Transfer('1',2)
