#pragma once
#include "npy.hpp"
#include <types.hpp>
#include <string>
#include <vector>

/**
 * @class Halo
 * @brief Owns read-only device-side data (cell AABBs, star/target positions) shared
 *        across ray-tracing kernels, and drives host-side chunked kernel dispatch.
 *
 * Memory-safe. Device buffers are RAII-managed by DeviceArray; no manual free needed.
 */

namespace Raytracer
{
	// Host-side, chunk-accumulated result of ray_trace_1. Indices are global into the
	// star (star_ind), target/cell-center (target_ind) and cell (cell_ind) buffers.
	struct Rt1HostResult
	{
		std::vector<double> dr;
		std::vector<uint32_t> target_ind;
		std::vector<uint32_t> cell_ind;
		std::vector<uint32_t> star_ind;
		size_t size() const { return dr.size(); }
	};

	class Halo
	{
	public:
		Halo() = default;
		Halo(bool cell_based, unsigned max_i, unsigned max_f) : _cell_based(cell_based), _max_ipos(max_i), _max_fpos(max_f) {};
		void init(const std::string data_path, const std::string halo_name);
		// Runs ray_trace_1 over all star/target chunks and returns the accumulated host result.
		const Rt1HostResult &rayTrace1();

		size_t numCells() const { return _ll.size(); }
		size_t numStars() const { return _spos.size(); }
		size_t numTargets() const { return _fpos.size(); }

	private:
		bool _cell_based = true;
		unsigned _max_ipos = 500;
		unsigned _max_fpos = 500;
		Raytrace1Result _rt1;	   // reused per-chunk device buffers
		Rt1HostResult _rt1_host;   // accumulated host-side results

		/*GPU-side raw data buffers*/
		DeviceArray<double3> _ll;
		DeviceArray<double3> _ur;
		DeviceArray<double3> _spos;
		DeviceArray<double3> _fpos; // per-cell centers = (ll + ur) / 2, used as ray targets
	};

}
