#include <rt1_kernels.cuh>
#include <cub/cub.cuh>
#include <math_constants.h>

namespace Raytracer
{

	__device__ bool slab_test(double3 ll, double3 ur, double3 r_or, double3 r_dir, double *t_enter = nullptr, double *t_exit = nullptr)
	{
		double tmin = 0.0;
		double tmax = CUDART_INF;
		// x
		double inv_dx = 1.0 / r_dir.x;
		double t0 = (ll.x - r_or.x) * inv_dx;
		double t1 = (ur.x - r_or.x) * inv_dx;
		tmin = fmax(tmin, fmin(t0, t1));
		tmax = fmin(tmax, fmax(t0, t1));
		if (tmax < tmin)
			return false;
		// y
		double inv_dy = 1.0 / r_dir.y;
		t0 = (ll.y - r_or.y) * inv_dy;
		t1 = (ur.y - r_or.y) * inv_dy;
		tmin = fmax(tmin, fmin(t0, t1));
		tmax = fmin(tmax, fmax(t0, t1));
		if (tmax < tmin)
			return false;
		// z
		double inv_dz = 1.0 / r_dir.z;
		t0 = (ll.z - r_or.z) * inv_dz;
		t1 = (ur.z - r_or.z) * inv_dz;
		tmin = fmax(tmin, fmin(t0, t1));
		tmax = fmin(tmax, fmax(t0, t1));
		bool hit = (tmin <= tmax) && (tmin <= 1.0) && (tmax >= 0.0);
		if (!hit)
			return false;
		/*update times for entrance and exit if in second phase*/
		if (t_enter)
			*t_enter = tmin;
		if (t_exit)
			*t_exit = tmax;
		return true;
	}

	// test every ray against every cell using the slab method. populate intersection count array for each ray
	__global__ void isectKernel(const double3 *ll, const double3 *ur, const double3 *ray_ipos, const double3 *ray_fpos,
								const size_t i_offset, const size_t f_offset, const size_t n_ipos, const size_t n_fpos,
								const size_t n_rays, const size_t n_cells, uint32_t *isect)
	{
		for (size_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rays; r += gridDim.x * blockDim.x)
		{
			uint32_t total = 0;
			size_t start_index = i_offset + (r % n_ipos);
			size_t end_index = f_offset + (r / n_ipos);
			double3 ray_start = ray_ipos[start_index];
			double3 ray_end = ray_fpos[end_index];
			double3 ray_dir = {ray_end.x - ray_start.x, ray_end.y - ray_start.y, ray_end.z - ray_start.z};
			// TODO: add shared memory loading in chunks
			for (size_t c = 0; c < n_cells; c++)
			{
				double3 lower_left = ll[c];
				double3 upper_right = ur[c];
				bool hit = slab_test(lower_left, upper_right, ray_start, ray_dir);
				if (hit)
					total++;
			}
			isect[r] = total;
		}
		return;
	}

	__global__ void rt1Kernel(const double3 *ll, const double3 *ur, const double3 *ray_ipos, const double3 *ray_fpos,
							  const size_t i_offset, const size_t f_offset, const size_t n_ipos, const size_t n_fpos,
							  const size_t n_rays, const size_t n_cells, const unsigned *offsets,
							  uint32_t *target_ind, uint32_t *cell_ind, uint32_t *star_ind, double *dr, bool cell_based)
	{
		for (size_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rays; r += gridDim.x * blockDim.x)
		{
			size_t start_index = i_offset + (r % n_ipos);
			size_t end_index = f_offset + (r / n_ipos);
			double3 ray_start = ray_ipos[start_index];
			double3 ray_end = ray_fpos[end_index];
			double3 ray_dir = {ray_end.x - ray_start.x, ray_end.y - ray_start.y, ray_end.z - ray_start.z};
			unsigned off_ind = offsets[r];
			for (size_t c = 0; c < n_cells; c++)
			{
				double t_enter;
				double t_exit;
				double3 lower_left = ll[c];
				double3 upper_right = ur[c];
				if (slab_test(lower_left, upper_right, ray_start, ray_dir, &t_enter, &t_exit))
				{
					t_enter = fmax(0.0, t_enter);
					if (!cell_based)
						t_exit = fmin(1.0, t_exit);
					double span = t_exit - t_enter;
					double dx = span * ray_dir.x, dy = span * ray_dir.y, dz = span * ray_dir.z;
					dr[off_ind] = sqrt(dx * dx + dy * dy + dz * dz);
					target_ind[off_ind] = (uint32_t)end_index;
					star_ind[off_ind] = (uint32_t)start_index;
					cell_ind[off_ind] = (uint32_t)c;
					off_ind++;
				}
			}
		}
		return;
	}

	__global__ void centersKernel(const double3 *ll, const double3 *ur, double3 *centers, size_t n_cells)
	{
		for (size_t c = blockIdx.x * blockDim.x + threadIdx.x; c < n_cells; c += gridDim.x * blockDim.x)
		{
			centers[c].x = 0.5 * (ll[c].x + ur[c].x);
			centers[c].y = 0.5 * (ll[c].y + ur[c].y);
			centers[c].z = 0.5 * (ll[c].z + ur[c].z);
		}
	}

	void computeCenters(const double3 *ll, const double3 *ur, double3 *centers, size_t n_cells)
	{
		if (n_cells == 0)
			return;
		const size_t num_blocks = (n_cells + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		centersKernel<<<num_blocks, THREADS_PER_BLOCK>>>(ll, ur, centers, n_cells);
		CUDA_CHECK(cudaGetLastError());
	}

	// Traces one host-chunked batch of rays (a sub-range of stars against a sub-range of targets).
	void rayTrace1(const double3 *ll, const double3 *ur, size_t n_cells,
				   const double3 *ray_ipos, size_t i_offset, size_t n_ipos,
				   const double3 *ray_fpos, size_t f_offset, size_t n_fpos,
				   Raytrace1Result &rt1, bool cell_based)
	{
		const size_t num_rays = n_ipos * n_fpos;
		if (num_rays == 0 || n_cells == 0)
		{
			rt1.dr.resize(0);
			rt1.ray_ind.target_ind.resize(0);
			rt1.ray_ind.cell_ind.resize(0);
			rt1.ray_ind.star_ind.resize(0);
			return;
		}
		DeviceArray<uint32_t> isect(num_rays);
		const size_t num_blocks = (num_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		// launch intersection kernel to generate per-ray intersection counts, because we need to know the total # of
		// intersections before launching the dr computation kernel
		isectKernel<<<num_blocks, THREADS_PER_BLOCK>>>(ll, ur, ray_ipos, ray_fpos, i_offset, f_offset, n_ipos, n_fpos,
													   num_rays, n_cells, isect.data());
		CUDA_CHECK(cudaGetLastError());
		DeviceArray<unsigned> offsets(num_rays + 1);
		void *d_temp = nullptr;
		size_t temp_bytes = 0;
		// compute number of temporary bytes needed for sum
		cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, isect.data(), offsets.data() + 1, num_rays);
		CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
		// inclusive prefix sum: each element i includes itself in the reduction along with all previous elements.
		// we shift the output by one index so offsets[r] is the exclusive prefix (per-ray write cursor), and
		// offsets[num_rays] holds the grand total.
		cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, isect.data(), offsets.data() + 1, num_rays);
		CUDA_CHECK(cudaFree(d_temp));
		// pull the total number of intersections back to the host so we can size the output buffers
		uint32_t total_isect = 0;
		CUDA_CHECK(cudaMemcpy(&total_isect, offsets.data() + num_rays, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		// resize ray_trace_1 result arrays
		rt1.ray_ind.target_ind.resize(total_isect);
		rt1.ray_ind.cell_ind.resize(total_isect);
		rt1.ray_ind.star_ind.resize(total_isect);
		rt1.dr.resize(total_isect);
		if (total_isect == 0)
			return;
		rt1Kernel<<<num_blocks, THREADS_PER_BLOCK>>>(ll, ur, ray_ipos, ray_fpos, i_offset, f_offset, n_ipos, n_fpos,
													 num_rays, n_cells, offsets.data(), rt1.ray_ind.target_ind.data(),
													 rt1.ray_ind.cell_ind.data(), rt1.ray_ind.star_ind.data(),
													 rt1.dr.data(), cell_based);
		CUDA_CHECK(cudaGetLastError());
	}

}
