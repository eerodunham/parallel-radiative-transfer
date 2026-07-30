#include <rt1_kernels.cuh>

namespace Raytracer
{

	__device__ bool slab_test(double3 ll, double3 ur, double3 r_or, double3 r_dir, float *t_enter = nullptr, float *t_exit = nullptr)
	{
		double tmin = 0.0f;
		double tmax = INFINITY;
		// x
		double inv_dx = 1.0f / r_dir.x;
		double t0 = (ll.x - r_or.x) * inv_dx;
		double t1 = (ur.x - r_or.x) * inv_dx;
		tmin = fmax(tmin, t0);
		tmax = fmin(tmax, t1);
		if (tmax < tmin)
			return false;
		// y
		double inv_dy = 1.0f / r_dir.y;
		t0 = (ll.y - r_or.y) * inv_dy;
		t1 = (ur.y - r_or.y) * inv_dy;
		tmin = fmax(tmin, t0);
		tmax = fmin(tmax, t1);
		if (tmax < tmin)
			return false;
		// z
		double inv_dz = 1.0f / r_dir.z;
		t0 = (ll.z - r_or.z) * inv_dz;
		t1 = (ur.z - r_or.z) * inv_dz;
		tmin = fmax(tmin, t0);
		tmax = fmin(tmax, t1);
		bool hit = (tmin <= tmax) && (tmin <= 1.0f)  && (tmax >= 0.0f);
		if(!hit) return false;
		/*update times for entrance and exit if in second phase*/
		if (t_enter)
			*t_enter = tmin;
		if (t_exit)
			*t_exit = tmax;
		return true;
	}

	// test every ray against every cell using the slab method. populate intersection count array for each ray
	__global__ void isectKernel(const double3 *ll, const double3 *ur, const double3 *ray_ipos, const double3 *ray_fpos, const size_t n_ipos, const size_t n_fpos, const size_t n_rays, const size_t n_cells, uint32_t *isect)
	{
		for (size_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rays; r += gridDim.x * blockDim.x)
		{
			uint32_t total = 0;
			int start_index = r % n_ipos;
			int end_index = r / n_ipos;
			double3 ray_start = ray_ipos[start_index];
			double3 ray_end = ray_fpos[end_index];
			double3 ray_dir = {ray_end.x - ray_start.x, ray_end.y - ray_start.y, ray_end.z - ray_start.z};
			// TODO: add shared memory loading in chunks
			for (int c = 0; c < n_cells; c++)
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

	__global__ void rt1Kernel(const double3 *ll, const double3 *ur, const double3 *ray_ipos, const double3 *ray_fpos, const size_t n_ipos, const size_t n_fpos, const size_t n_rays, const size_t n_cells, const unsigned *offsets,
                           uint32_t *target_ind, uint32_t *cell_ind, uint32_t *star_ind,
                           double *dr) {
		for (size_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rays; r += gridDim.x * blockDim.x)
		{
			uint32_t start_index = r % n_ipos;
			uint32_t end_index = r / n_ipos;
			double3 ray_start = ray_ipos[start_index];
			double3 ray_end = ray_fpos[end_index];
			double3 ray_dir = {ray_end.x - ray_start.x, ray_end.y - ray_start.y, ray_end.z - ray_start.z};
			unsigned off_ind = offsets[r];
			for(int c = 0; c < n_cells; c++) {
				float t_enter; float t_exit;
				double3 lower_left = ll[c];
				double3 upper_right = ur[c];
				if(slab_test(lower_left, upper_right, ray_start, ray_dir, &t_enter, &t_exit)) {
					t_enter = fmaxf(0.0f, t_enter);
					double dx = (double)(t_exit-t_enter)*ray_dir.x, dy = (double)(t_exit-t_enter)*ray_dir.y, dz = (double)(t_exit-t_enter)*ray_dir.z;
					dr[off_ind] = sqrt(dx*dx + dy*dy + dz*dz);
					target_ind[off_ind] = end_index;
					star_ind[off_ind] = start_index;
					cell_ind[off_ind] = c;
					off_ind++;
				}
			}
		}
		return;
	}
	//chunked on host-side, then passed to device arrays.
	void rayTrace1(AABB &aabb, Raytrace1Result &rt1, const DeviceArray<double3> &ray_ipos, const DeviceArray<double3> &ray_fpos, GridData grid) {
		const size_t num_rays = ray_ipos.size() * ray_fpos.size();
		const size_t n_cells = grid.num_cells;
		DeviceArray<uint32_t> isect(num_rays);
		const size_t num_blocks = (num_rays + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		//launch intersection kernel to generate per-cell intersection counts, because we need to know the total # of intersections before launching
		//the dr computation kernel
		isectKernel<<<num_blocks, THREADS_PER_BLOCK>>>(aabb.ll.data(), aabb.ur.data(), ray_ipos.data(), ray_fpos.data(), ray_ipos.size(), ray_fpos.size(), num_rays, n_cells, isect.data());
		CUDA_CHECK(cudaGetLastError());
		DeviceArray<unsigned> offsets(num_rays+1);
		void  *d_temp = nullptr; size_t temp_bytes = 0;
		//compute number of temporary bytes needed for sum
		cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, isect.data(), offsets.data()+1, num_rays);
		CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
		//inclusive prefix sum: each element i includes itself in the reduction along with all previous elements
		//we shift by one index so the indices are also the per-ray offsets into the intersection buffer!
		cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, isect.data(), offsets.data()+1, num_rays);
		CUDA_CHECK(cudaFree(d_temp));
		//now get the value at last index of the intersection array. this is the total number of intersections
		uint32_t total_isect = 0;
		CUDA_CHECK(cudaMemcpy(&total_isect, offsets.data()+num_rays, sizeof(int), cudaMemcpyDeviceToHost));
		//resize ray_trace_1 result arrays
		rt1.ray_ind.target_ind.resize(total_isect);
		rt1.ray_ind.cell_ind.resize(total_isect);
		rt1.ray_ind.star_ind.resize(total_isect);
		rt1.dr.resize(total_isect);
		rt1Kernel<<<num_blocks, THREADS_PER_BLOCK>>>(aabb.ll.data(), aabb.ur.data(), ray_ipos.data(), ray_fpos.data(), ray_ipos.size(), ray_fpos.size(), num_rays, n_cells, offsets.data(), rt1.ray_ind.target_ind.data(),
		rt1.ray_ind.cell_ind.data(), rt1.ray_ind.star_ind.data(), rt1.dr.data());
		CUDA_CHECK(cudaGetLastError());
	}

}