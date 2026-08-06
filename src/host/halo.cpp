#include "halo.h"
#include <rt1_kernels.cuh>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace Raytracer
{
	namespace
	{
		// Stage a [N,3] float64 npy array through pinned host memory into a device double3 buffer.
		void uploadVec3(DeviceArray<double3> &dst, const npy::npy_data<double> &src, const std::string &name)
		{
			if (src.shape.size() != 2 || src.shape[1] != 3)
				throw std::runtime_error("Halo::init: expected [N,3] array for " + name);
			const size_t n = src.shape[0];
			dst.resize(n);
			if (n == 0)
				return;
			double3 *pinned = nullptr;
			CUDA_CHECK(cudaHostAlloc(&pinned, n * sizeof(double3), cudaHostAllocDefault));
			std::memcpy(pinned, src.data.data(), n * sizeof(double3));
			CUDA_CHECK(cudaMemcpy(dst.data(), pinned, n * sizeof(double3), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaFreeHost(pinned));
		}
	} 

	void Halo::init(const std::string data_path, const std::string halo_name)
	{
		const fs::path halo_dir = fs::path(data_path) / halo_name;
		if (!fs::exists(halo_dir) || !fs::is_directory(halo_dir))
		{
			throw std::runtime_error(
				"Halo::init: halo timestep data directory not found: " + halo_dir.string());
		}

		const std::array<std::string, 3> required_files = {
			"grid_ll.npy", "grid_ur.npy", "stars_positions.npy"};

		for (const auto &fname : required_files)
		{
			const fs::path fpath = halo_dir / fname;
			if (!fs::exists(fpath))
			{
				throw std::runtime_error(
					"Halo::init: missing required data file: " + fpath.string());
			}
		}
		try
		{
			npy::npy_data<double> ll_data = npy::read_npy<double>((halo_dir / "grid_ll.npy").string());
			npy::npy_data<double> ur_data = npy::read_npy<double>((halo_dir / "grid_ur.npy").string());
			npy::npy_data<double> spos_data = npy::read_npy<double>((halo_dir / "stars_positions.npy").string());

			uploadVec3(_ll, ll_data, "grid_ll.npy");
			uploadVec3(_ur, ur_data, "grid_ur.npy");
			uploadVec3(_spos, spos_data, "stars_positions.npy");

			if (_ll.size() != _ur.size())
				throw std::runtime_error("Halo::init: grid_ll and grid_ur cell counts differ");

			// Derive per-cell target positions (cell centers) on the device.
			_fpos.resize(_ll.size());
			computeCenters(_ll.data(), _ur.data(), _fpos.data(), _ll.size());
			CUDA_CHECK(cudaDeviceSynchronize());
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(
				std::string("Halo::init: failed to read .npy files in ") +
				halo_dir.string() + ": " + e.what());
		}
	}

	const Rt1HostResult &Halo::rayTrace1()
	{
		_rt1_host = Rt1HostResult{};
		const size_t n_cells = _ll.size();
		const size_t n_stars = _spos.size();
		const size_t n_targets = _fpos.size();

		for (size_t i0 = 0; i0 < n_stars; i0 += _max_ipos)
		{
			const size_t ni = std::min<size_t>(_max_ipos, n_stars - i0);
			for (size_t f0 = 0; f0 < n_targets; f0 += _max_fpos)
			{
				const size_t nf = std::min<size_t>(_max_fpos, n_targets - f0);
				Raytracer::rayTrace1(_ll.data(), _ur.data(), n_cells,
									 _spos.data(), i0, ni,
									 _fpos.data(), f0, nf,
									 _rt1, _cell_based);

				const size_t k = _rt1.dr.size(); // total intersections for this chunk
				if (k == 0)
					continue;
				const size_t old = _rt1_host.dr.size();
				_rt1_host.dr.resize(old + k);
				_rt1_host.target_ind.resize(old + k);
				_rt1_host.cell_ind.resize(old + k);
				_rt1_host.star_ind.resize(old + k);
				_rt1.dr.copyToHost(_rt1_host.dr.data() + old, k);
				_rt1.ray_ind.target_ind.copyToHost(_rt1_host.target_ind.data() + old, k);
				_rt1.ray_ind.cell_ind.copyToHost(_rt1_host.cell_ind.data() + old, k);
				_rt1.ray_ind.star_ind.copyToHost(_rt1_host.star_ind.data() + old, k);
			}
		}
		return _rt1_host;
	}
}
