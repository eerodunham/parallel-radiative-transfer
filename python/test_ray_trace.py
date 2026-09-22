import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import yt

from spectra_solver_mk2 import gpu_ray_trace_1, gpu_ray_trace_4

def test_raytrace_mock(dataset_path, halo_tree_path, stars_path, halo_id, timestep, n_ipos, n_fpos, n_nu, n_time_bins):
    '''
        Tests/Times rt1 and rt4 using mock data.
        Actual data includes cell positions, star positions, and densities
        Feaux data is generated for:
            nu 
            mu
            spectra_times
            opacity
            temporal indexing arrays
            bool_past
    '''
    print(f"Loading yt dataset: {dataset_path}")
    ds = yt.load(dataset_path)
    halo_tree = np.load(halo_tree_path, allow_pickle=True).tolist()
    stars = np.load(stars_path, allow_pickle=True).tolist()
    id = str(halo_id)
    center = halo_tree[id][timestep]['Halo_Center']
    radius = halo_tree[id][timestep]['Halo_Radius'] * 1.1
    reg = ds.sphere(center, radius)

    #data loaded from ds
    dx_cgs = reg['dx'].in_units("cm").v[:n_fpos]
    x = reg[("gas", "x")].in_units("cm").v[:n_fpos]
    y = reg[("gas", "y")].in_units("cm").v[:n_fpos]
    z = reg[("gas", "z")].in_units("cm").v[:n_fpos]
    ll = np.column_stack([x - dx_cgs/2.0, y - dx_cgs/2.0, z - dx_cgs/2.0])
    ur = np.column_stack([x + dx_cgs/2.0, y + dx_cgs/2.0, z + dx_cgs/2.0])
    fpos = np.column_stack([x, y, z])
    den = reg[("gas", "density")].in_units("g/cm**3").v[:n_fpos]
    ipos = (stars[id][timestep]['positions2'] * ds.length_unit.in_units("cm").v)[:n_ipos]
    #mock star position tensor (light travel time ignored)
    ipos_3d = np.repeat(ipos[np.newaxis, :, :], n_fpos, axis=0)
    print(fpos.shape, ipos.shape)
    target_cells = np.arange(n_fpos, dtype=np.int64)


    #can use opacity from plothype
    ''' MOCK DATA '''

    #mock freq bins
    nu = np.logspace(15, 17, n_nu)
    #mock molecular weight
    mu = np.full(n_fpos, 1.22)
    #mock spectra times
    spectra_times = np.random.uniform(1e-22, 1e-20, size=(n_time_bins, n_ipos, n_nu))
    #mock opacity lookup
    opacity = 10**(np.random.uniform(-19, -17, size=(n_fpos, n_time_bins, n_nu)))
    #mock temporal indexing arrays
    source_lo = np.random.randint(0, n_time_bins - 1, size=(n_fpos, n_ipos), dtype=np.int64)
    source_w = np.random.uniform(0.0, 1.0, size=(n_fpos, n_ipos))
    bool_past = np.ones((n_fpos, n_ipos), dtype=bool)
    

    print("\n--- Starting Ray Trace Execution ---")
    start_rt1 = cp.cuda.Event()
    end_rt1 = cp.cuda.Event()
    start_rt4 = cp.cuda.Event()
    end_rt4 = cp.cuda.Event()

    start_rt1.record()
    dr, ray_ind, ray_fraction = gpu_ray_trace_1(ll, ur, dx_cgs, ipos_3d, fpos)
    end_rt1.record()
    n_rays = len(dr)
    #mock redshift (default to 0)
    redshift = np.zeros(n_rays, dtype=np.float64)
    unique_cells, ray_cell_local = np.unique(ray_ind[:, 1], return_inverse=True)
    ray_lo = np.random.randint(0, n_time_bins - 1, size=n_rays, dtype=np.int32)
    ray_hi = (ray_lo + 1).astype(np.int32)
    ray_w = np.random.uniform(0.0, 1.0, size=n_rays).astype(np.float64)

    start_rt4.record()
    final_intensity = gpu_ray_trace_4(ray_ind, ray_cell_local, dr, ipos_3d, 
                                      fpos, nu, den, mu, opacity, redshift, 
                                      target_cells, spectra_times,
                                      ray_hi, ray_lo, ray_w, 
                                      source_lo, source_w, bool_past)
    end_rt4.record()

    end_rt4.synchronize()
    rt1_ms = cp.cuda.get_elapsed_time(start_rt1, end_rt1)
    rt4_ms = cp.cuda.get_elapsed_time(start_rt4, end_rt4)
    print(f"RT1 Execution Time: {rt1_ms:.3f} ms")
    print(f"RT4 Execution Time: {rt4_ms:.3f} ms")
    print(f"Total GPU Runtime:  {rt1_ms + rt4_ms:.3f} ms")

    unattenuated_total_spectrum = np.sum(spectra_times[-1, :, :], axis=0)
    attenuated_total_spectrum = np.sum(final_intensity, axis=0)
    print(attenuated_total_spectrum)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(nu, unattenuated_total_spectrum, label="Initial Total Source Spectrum (Before)", color="crimson", lw=2)
    ax.loglog(nu, attenuated_total_spectrum, label="Integrated Target Cell Spectrum (After)", color="navy", linestyle="--", lw=2)

    ax.set_xlabel(r"Frequency $\nu$ [Hz]")
    ax.set_ylabel(r"Integrated Intensity $I_\nu$ [ergs s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    ax.set_title("Radiative Transfer Spectrum Comparison (Before vs. After)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig("rt_spectra_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    dataset_path = "../data/box_3_z_1/DD0679/output_0679"
    halo_tree = "../data/halotree_2020_final.npy"
    star_list = "../data/starlists_2020.npy"
    halo_id = 0
    timestep = 0
    n_ipos = 2
    n_fpos = 2
    n_nu = 10
    n_time_bins = 5
    test_raytrace_mock(dataset_path, halo_tree, star_list, halo_id, timestep, n_ipos, n_fpos, n_nu, n_time_bins)
    print("TEST COMPLETE")

#chunking is 500 ipos 50 fpos 2000 nu

#rt1 is possible target for fp32