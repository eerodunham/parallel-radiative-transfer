#include "npy.hpp"
#include <types.hpp>

/**
 * @class Raytracer
 * @brief Owns temporary device-side buffers for all ray-tracing functions
 *
 * Memory-safe. No data requires manual de-allocation
 */

namespace Raytracer
{
	class Halo
	{
	public:
		Halo() = default;
		Halo(bool cell_based, unsigned max_i, unsigned max_f) : _cell_based(cell_based), _max_ipos(max_i), _max_fpos(max_f) {};
		void init(const std::string data_path, const std::string halo_name);
	void rayTrace1() private : bool _cell_based = true;
		unsigned _max_ipos = 500;
		unsigned _max_fpos = 500;
		Raytrace1Result _rt1;

		/*GPU-side raw data buffers*/
		DeviceArray<double3> _ll;
		DeviceArray<double3> _ur;
		DeviceArray<double3> _spos;
	};

}