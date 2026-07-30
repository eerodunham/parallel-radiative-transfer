#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <types.hpp>

#define THREADS_PER_BLOCK 256
namespace Raytracer
{
    void rayTrace1(AABB &aabb, Raytrace1Result &rt1, const DeviceArray<double3> &ray_ipos, const DeviceArray<double3> &ray_fpos, GridData grid);
}