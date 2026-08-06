#include "halo.h"
#include <rt1_kernels.cuh>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace Raytracer
{
	void Halo::rayTrace1()
	{
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

			size_t num_cells = ll_data.shape[0];
			size_t num_stars = spos_data.shape[0];

			double3 *h_ll_pinned = nullptr;
			cudaHostAlloc(&h_ll_pinned, num_cells * sizeof(double3), cudaHostAllocDefault);
			std::memcpy(h_ll_pinned, ll_data.data.data(), num_cells * sizeof(double3));
			_ll.resize(num_cells);
			cudaMemcpy(_ll.data(), h_ll_pinned, num_cells * sizeof(double3), cudaMemcpyHostToDevice);

			double3 *h_ur_pinned = nullptr;
			cudaHostAlloc(&h_ur_pinned, num_cells * sizeof(double3), cudaHostAllocDefault);
			std::memcpy(h_ur_pinned, ur_data.data.data(), num_cells * sizeof(double3));
			_ur.resize(num_cells);
			cudaMemcpy(_ur.data(), h_ur_pinned, num_cells * sizeof(double3), cudaMemcpyHostToDevice);

			double3 *h_spos_pinned = nullptr;
			cudaHostAlloc(&h_spos_pinned, num_stars * sizeof(double3), cudaHostAllocDefault);
			std::memcpy(h_spos_pinned, ll_data.data.data(), num_stars * sizeof(double3));
			_spos.resize(num_stars);
			cudaMemcpy(_spos.data(), h_spos_pinned, num_stars * sizeof(double3), cudaMemcpyHostToDevice);
		}
		catch (const std::exception &e)
		{
			throw std::runtime_error(
				std::string("Halo::init: failed to read .npy files in ") +
				halo_dir.string() + ": " + e.what());
		}
	}
}