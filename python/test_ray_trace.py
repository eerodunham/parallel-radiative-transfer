import time
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import yt

from spectra_solver_mk2 import gpu_ray_trace_1, gpu_ray_trace_4

def test_raytrace_mock(dataset_path, n_ipos, n_fpos, n_nu, n_time_bins):
    print(f"Loading yt dataset: {dataset_path}")
    ds = yt.load(dataset_path)
    ad = ds.all_data()

    #data loaded from ds
    dx_cgs = ad[("gas", "dx")].in_units("cm").v[:n_fpos]
    x = ad[("gas", "x")].in_units("cm").v[:n_fpos]
    y = ad[("gas", "y")].in_units("cm").v[:n_fpos]
    z = ad[("gas", "z")].in_units("cm").v[:n_fpos]
    ll = np.column_stack([x - dx_cgs/2.0, y - dx_cgs/2.0, z - dx_cgs/2.0])
    ur = np.column_stack([x + dx_cgs/2.0, y + dx_cgs/2.0, z + dx_cgs/2.0])
    fpos = np.column_stack([x, y, z])
    den = ad[("gas", "density")].in_units("g/cm**3").v[:n_fpos]
    ipos = ad[("io", "particle_position")].in_units("cm").v[:n_ipos]
    target_cells = np.arange(n_fpos, dtype=np.int64)

    #mock data

    #mock freq bins
    nu = np.logspace(15, 17, n_nu)
    #mock molecular weight
    mu = np.full(n_fpos, 1.22)
    #mock spectra times
    spectra_times = np.random.uniform(1e-22, 1e-20, size=(n_time_bins, n_ipos, n_nu))
    #mock opacity lookup
    opacity = np.random.uniform(10.0, 100.0, size=(n_fpos, n_time_bins, n_nu))
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
    dr, ray_ind, ray_fraction = gpu_ray_trace_1(ll, ur, dx_cgs, ipos, fpos)
    end_rt1.record()
    n_rays = len(dr)
    #mock redshift (default to 0)
    redshift = np.zeros(n_rays, dtype=np.float64)
    unique_cells, ray_cell_local = np.unique(ray_ind[:, 1], return_inverse=True)
    ray_lo = np.random.randint(0, n_time_bins - 1, size=n_rays, dtype=np.int32)
    ray_hi = (ray_lo + 1).astype(np.int32)
    ray_w = np.random.uniform(0.0, 1.0, size=n_rays).astype(np.float64)

    start_rt4.record()
    final_intensity = gpu_ray_trace_4(ray_ind, ray_cell_local, dr, ipos, 
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

if __name__ == "__main__":
    test_raytrace_mock("MOCK", n_ipos=100, n_fpos=100, n_nu=1000, n_time_bins=10)
