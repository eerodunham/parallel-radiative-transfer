#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>

#define CUDA_CHECK(err)                                        \
	if (err != cudaSuccess)                                    \
	{                                                          \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
				  << " at line " << __LINE__ << std::endl;     \
		exit(EXIT_FAILURE);                                    \
	}

// Basic device array wrapper
// Taken from VEG-INR repo (eerodunham)
namespace Raytracer
{
	template <typename T>

	struct DeviceArray
	{
		DeviceArray() = default;
		DeviceArray(const DeviceArray &) = delete;
		DeviceArray &operator=(const DeviceArray &) = delete;
		DeviceArray(DeviceArray &&other) noexcept : _data(other._data), _size(other._size)
		{
			other._data = nullptr;
			other._size = 0;
		}
		DeviceArray &operator=(DeviceArray &&other) noexcept
		{
			if (this != &other)
			{
				if (_data)
					CUDA_CHECK(cudaFree(_data));
				_data = other._data;
				_size = other._size;
				other._data = nullptr;
				other._size = 0;
			}
			return *this;
		}

		DeviceArray(size_t n) : _size(n)
		{
			CUDA_CHECK(cudaMalloc((void **)&_data, _size * sizeof(T)));
			CUDA_CHECK(cudaMemset(_data, (const int)0, _size * sizeof(T)));
		}
		DeviceArray(size_t n, const int init) : _size(n)
		{
			CUDA_CHECK(cudaMalloc((void **)&_data, _size * sizeof(T)));
			CUDA_CHECK(cudaMemset(_data, init, _size * sizeof(T)));
		}
		void copyToDevice(T *host, size_t n)
		{
			assert(n <= _size);
			CUDA_CHECK(cudaMemcpy(_data, host, n * sizeof(T), cudaMemcpyHostToDevice));
		}
		void copyToHost(T *host, size_t n)
		{
			if (n > _size)
				n = _size;
			CUDA_CHECK(cudaMemcpy(host, _data, n * sizeof(T), cudaMemcpyDeviceToHost));
		}
		void resize(size_t newsize)
		{
			if (_data)
				CUDA_CHECK(cudaFree(_data));
			CUDA_CHECK(cudaMalloc((void **)&_data, newsize * sizeof(T)));
			CUDA_CHECK(cudaMemset(_data, 0, newsize * sizeof(T)));
			_size = newsize;
		}
		~DeviceArray()
		{
			CUDA_CHECK(cudaFree(_data));
		}

		T *data() const { return _data; }
		size_t size() const { return _size; }

	private:
		T *_data = nullptr;
		size_t _size = 0;
	};

	struct GridData
	{
		size_t num_stars = 0;
		size_t num_cells = 0;
		size_t num_wav = 70000;
	};
	struct AABB
	{
		DeviceArray<double3> ll;
		DeviceArray<double3> ur;
	};
	// Data for all Ray Segments
	struct RayIndex
	{
		DeviceArray<uint32_t> target_ind;
		DeviceArray<uint32_t> cell_ind;
		DeviceArray<uint32_t> star_ind;
	};
	// Result of calling ray_trace_1
	struct Raytrace1Result
	{
		DeviceArray<double> dr;
		RayIndex ray_ind;
	};
}