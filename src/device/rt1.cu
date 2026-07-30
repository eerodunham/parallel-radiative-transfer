#include <rt1_kernels.cuh>

namespace Raytracer
{

	__device__ bool slab_test(float3 ll, float3 ur, float3 r_or, float3 r_dir, float *t_enter = nullptr, float *t_exit = nullptr)
	{
		float tmin = 0.0f;
		float tmax = 1.0f;
		// x
		float inv_dx = 1.0f / r_dir.x;
		float t0 = (ll.x - r_or.x) * inv_dx;
		float t1 = (ur.x - r_or.x) * inv_dx;
		if (inv_dx < 0.0f)
		{
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tmin = fmaxf(tmin, t0);
		tmax = fminf(tmax, t1);
		if (tmax < tmin)
			return false;

		// y
		float inv_dy = 1.0f / r_dir.y;
		t0 = (ll.y - r_or.y) * inv_dy;
		t1 = (ur.y - r_or.y) * inv_dy;
		if (inv_dy < 0.0f)
		{
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tmin = fmaxf(tmin, t0);
		tmax = fminf(tmax, t1);
		if (tmax < tmin)
			return false;

		// z
		float inv_dz = 1.0f / r_dir.z;
		t0 = (ll.z - r_or.z) * inv_dz;
		t1 = (ur.z - r_or.z) * inv_dz;
		if (inv_dz < 0.0f)
		{
			float tmp = t0;
			t0 = t1;
			t1 = tmp;
		}
		tmin = fmaxf(tmin, t0);
		tmax = fminf(tmax, t1);
		if (tmax < tmin)
			return false;
		/*update times for entrance and exit if in second phase*/
		if (t_enter)
			*t_enter = tmin;
		if (t_exit)
			*t_exit = tmax;
		return true;
	}

	// test every ray against every cell using the slab method. populate pre-sized boolean array with 1 if intersection occurs
	__global__ void isectKernel(const float3 *ll, const float3 *ur, const int2 *ray_ids, const float3 *ray_ipos, const float3 *ray_fpos, const int n_rays, const int n_cells, int *isect)
	{
		for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rays; r += gridDim.x * blockDim.x)
		{
			int total = 0;
			int2 ray_id = ray_ids[r];
			float3 ray_start = ray_ipos[ray_id.x];
			float3 ray_end = ray_fpos[ray_id.y];
			float3 ray_dir = {ray_end.x - ray_start.x, ray_end.y - ray_start.y, ray_end.z - ray_start.z};
			// TODO: add shared memory loading in chunks
			for (int c = 0; c < n_cells; c++)
			{
				float3 lower_left = ll[c];
				float3 upper_right = ur[c];
				bool hit = slab_test(lower_left, upper_right, ray_start, ray_dir);
				if (hit)
					total++;
			}
			isect[r] = total;
		}
		return;
	}

}