import argparse
import os
import sys
import numpy as np
import h5py
try:
    import yt
except ImportError:
    yt = None
INT_DTYPE = np.int32
FLOAT_DTYPE= np.float64

def _write_dataset(h5file, name, arr, dtype):
    arr = np.ascontiguousarray(arr, dtype=dtype)
    ds = h5file.create_dataset(name, data=arr, dtype=dtype)
    ds.attrs["shape"] = np.array(arr.shape, dtype=np.int64)
    ds.attrs["dtype"] = np.dtype(dtype).name
    return ds

class Exporter:
    def __init__(self, data_path, timestep=0, halo_ver=2020, test=True):
        self.data_path = data_path
        self.timestep = timestep
        self.halo_ver = halo_ver
        self.test = test
        print(test)
        self._load_halo()
        self._load_ds()
        self._load_fields()
        self._load_stars()
        self._load_table()
        #testing mode ignores dust values
        if not test:
            self._load_dust()
        else:
            self.emiss_dust_0 = None
            self.dust_nrg = None
            self.chisdust_0 = None
            self.chivdust_0 = None
    def _p(self, *parts):
        return os.path.join(self.data_path, *parts)
    def _load_halo(self):
        halofile = self._p("halotree_%s_final.npy" % self.halo_ver)
        halo = np.load(halofile, allow_pickle=True).tolist()
        self.halo_rad = halo["0"][5]["Halo_Radius"]
        self.halo_center = np.asarray(halo["0"][5]["Halo_Center"], dtype=FLOAT_DTYPE)
    def _load_ds(self):
        if yt is None:
            raise RuntimeError("yt is required to load EnzoGrid datasets.")
        ds_path_1 = self._p("pfs_allsnaps_%s.txt" % self.halo_ver)
        file_list = np.loadtxt(ds_path_1, dtype=str)[:, 0]
        ds_file = file_list[self.timestep]
        #append to global path if not absolute
        if not os.path.isabs(ds_file):
            ds_file = os.path.join(self.data_path, ds_file)
        self.ds = yt.load(ds_file)
        radius = self.ds.length_unit.in_units("cm") * self.halo_rad
        self.reg = self.ds.box(
            (self.halo_center - self.halo_rad), (self.halo_center + self.halo_rad)
        )
        self.ll_box = (self.halo_center - radius.v).astype(FLOAT_DTYPE)
        self.ur_box = (self.halo_center + radius.v).astype(FLOAT_DTYPE)
    def _load_fields(self):
        reg=self.reg
        ll_x = (reg["x"] - reg["dx"] / 2).in_units("cm").v
        ll_y = (reg["y"] - reg["dy"] / 2).in_units("cm").v
        ll_z = (reg["z"] - reg["dz"] / 2).in_units("cm").v
        self.ll = np.column_stack((ll_x, ll_y, ll_z)).astype(FLOAT_DTYPE)
        v_x = reg["velocity_x"].in_units("cm/s").v
        v_y = reg["velocity_y"].in_units("cm/s").v
        v_z = reg["velocity_z"].in_units("cm/s").v
        self.vel = np.column_stack((v_x, v_y, v_z)).astype(FLOAT_DTYPE)
        self.dds = reg["dx"].in_units("cm").v.astype(FLOAT_DTYPE)
        self.ur = (self.ll + self.dds[:, np.newaxis]).astype(FLOAT_DTYPE)
        self.temps = reg["temperature"].in_units("K").v.astype(FLOAT_DTYPE)
        self.metals = reg["metallicity"].in_units("Zsun").v.astype(FLOAT_DTYPE)
        self.den = np.array(
            (reg["HI_Density"] + reg["H2I_Density"] + reg["H2II_Density"])
            .in_units("g/cm**3")
            .v,
            dtype=FLOAT_DTYPE,
        )
    def _load_stars(self):
        stars_file = self._p("starlists_%s.npy" % self.halo_ver)
        stars = np.load(stars_file, allow_pickle=True).tolist()
        length_cm = self.ds.length_unit.in_units("cm").v
        self.spos = (stars["0"][self.timestep]["positions2"] * length_cm).astype(FLOAT_DTYPE)
        self.svels = np.asarray(stars["0"][self.timestep]["vels2"], dtype=FLOAT_DTYPE)
    def _load_table(self):
        plotd_file = self._p("plothype0_52.npy")
        plotd = np.load(plotd_file, allow_pickle=True).tolist()
        self.chivmet_0 = np.asarray(plotd["chivmet"], dtype=FLOAT_DTYPE)
        self.chismet_0 = np.asarray(plotd["chismet"], dtype=FLOAT_DTYPE)
        self.chivhe = np.asarray(plotd["chivhe"], dtype=FLOAT_DTYPE)
        self.chishe = np.asarray(plotd["chishe"], dtype=FLOAT_DTYPE)
        self.nu = np.asarray(plotd["nu"], dtype=FLOAT_DTYPE)

        temp_table = np.asarray(plotd["temp"], dtype=FLOAT_DTYPE)
        i_temp = np.searchsorted(temp_table, self.temps)
        i_temp = np.minimum(i_temp, len(temp_table) - 1)
        self.i_temp = i_temp.astype(INT_DTYPE)
    def _load_dust(self):
        from hyperion.dust import SphericalDust
        if SphericalDust is None:
            raise RuntimeError("Hyperion is required to load dust data. Run in testing mode to ignore dust.")
        dust_file = self._p("hyperion-dust-0.1.0", "dust_files", "d03_4.0_4.0_A.hdf5")
        d = SphericalDust(dust_file)
        nu_d = d.optical_properties.nu
        kdust = np.interp(self.nu, nu_d, d.optical_properties.chi).T
        alb_dust = np.interp(self.nu, nu_d, d.optical_properties.albedo).T
        chisdust = alb_dust * kdust
        chivdust = kdust - chisdust
        self.emiss_dust_0 = np.array(
            [np.interp(self.nu, d.emissivities.nu, np.array(d.emissivities.jnu)[:, i]) for i in range(len(d.emissivities.jnu[0]))],
            dtype=FLOAT_DTYPE,
        )
        self.dust_nrg = np.asarray(d.emissivities.var, dtype=FLOAT_DTYPE)
        self.chisdust_0 = chisdust[np.newaxis, :].astype(FLOAT_DTYPE)
        self.chivdust_0 = chivdust[np.newaxis, :].astype(FLOAT_DTYPE)
    def _export(self, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        out_file = os.path.join(dump_path, "halo_t%03d.h5" % self.timestep)
        with h5py.File(out_file, "w") as f:
            f.attrs["timestep"] = self.timestep
            f.attrs["halo_version"] = self.halo_ver
            f.attrs["n_cells"] = self.ll.shape[0]
            f.attrs["n_stars"] = self.spos.shape[0]
            f.attrs["n_nu"] = self.nu.shape[0]
            grid = f.create_group("grid")
            _write_dataset(grid, "ll", self.ll, FLOAT_DTYPE)
            _write_dataset(grid, "ur", self.ur, FLOAT_DTYPE)
            _write_dataset(grid, "dds", self.dds, FLOAT_DTYPE)
            _write_dataset(grid, "velocity", self.vel, FLOAT_DTYPE)
            _write_dataset(grid, "temperature", self.temps, FLOAT_DTYPE)
            _write_dataset(grid, "metallicity", self.metals, FLOAT_DTYPE)
            _write_dataset(grid, "density", self.den, FLOAT_DTYPE)
            _write_dataset(grid, "i_temp", self.i_temp, INT_DTYPE)
            _write_dataset(grid, "ll_box", self.ll_box, FLOAT_DTYPE)
            _write_dataset(grid, "ur_box", self.ur_box, FLOAT_DTYPE)
            stars = f.create_group("stars")
            _write_dataset(stars, "positions", self.spos, FLOAT_DTYPE)
            _write_dataset(stars, "velocities", self.svels, FLOAT_DTYPE)
            chi = f.create_group("chi_table")
            _write_dataset(chi, "nu", self.nu, FLOAT_DTYPE)
            _write_dataset(chi, "chishe", self.chishe, FLOAT_DTYPE)
            _write_dataset(chi, "chismet_0", self.chismet_0, FLOAT_DTYPE)
            _write_dataset(chi, "chivhe", self.chivhe, FLOAT_DTYPE)
            _write_dataset(chi, "chivmet_0", self.chivmet_0, FLOAT_DTYPE)
            if self.emiss_dust_0 is not None:
                dust = f.create_group("dust")
                _write_dataset(dust, "emiss_dust_0", self.emiss_dust_0, FLOAT_DTYPE)
                _write_dataset(dust, "dust_nrg", self.dust_nrg, FLOAT_DTYPE)
                _write_dataset(dust, "chisdust_0", self.chisdust_0, FLOAT_DTYPE)
                _write_dataset(dust, "chivdust_0", self.chivdust_0, FLOAT_DTYPE)
        print("Wrote %s" % out_file)
        return out_file

def main():
    parser = argparse.ArgumentParser(
        description="Export EnzoGrid data to HDF5 for the CUDA radiative transfer solver to ingest."
    )
    parser.add_argument("data_path", type=str, help="Path to source data directory. Directory must contain a halo tree and masses file" \
	", plothype file and starlists, all as .npy files. Must also contain a pfs_allsnaps .txt file.")
    parser.add_argument("dump_path", type=str, help="Directory to write exported raw data files (in HDF5 format) into.")
    parser.add_argument("--timestep", type=int, default=0, help="Timestep index to load. Default: 0")
    parser.add_argument("--halo-version", type=int, default=2020, help="Halo catalog version year. Default: 2020")
    parser.add_argument("--test", type=bool, default=True, help="Toggles testing mode. Testing mode skips exporting dust data. Default: true")
    args = parser.parse_args()
    if not os.path.isdir(args.data_path):
        print("Error: data_path %s does not exist." % args.data_path, file=sys.stderr)
        sys.exit(1)
    exporter = Exporter(
        data_path=args.data_path,
        timestep=args.timestep,
        halo_ver=args.halo_version,
        test=args.test
    )
    exporter._export(args.dump_path)

if __name__ == "__main__":
    main()

        