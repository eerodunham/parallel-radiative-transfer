#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <types.hpp>

// NOTE: <cub/cub.cuh> is intentionally NOT included here so this header stays
// includable from host-compiled translation units (e.g. halo.cpp). CUB is only
// needed inside rt1.cu, which includes it directly.

#define THREADS_PER_BLOCK 256
namespace Raytracer
{
	// Fill `centers[c] = (ll[c] + ur[c]) / 2` for every cell. Used to produce the
	// per-cell target (final_pos) buffer that ray_trace_1 casts rays toward.
	void computeCenters(const double3 *ll, const double3 *ur, double3 *centers, size_t n_cells);

	// Cast a ray from each star (a chunk of `ray_ipos`) to each target (a chunk of
	// `ray_fpos`), test it against all `n_cells` cells, and accumulate one segment
	// length `dr` plus the (target, cell, star) indices per intersection.
	//
	// `ray_ipos`/`ray_fpos` are the full device buffers; only the sub-ranges
	// [i_offset, i_offset+n_ipos) and [f_offset, f_offset+n_fpos) are traced this
	// launch. Recorded star/target indices are GLOBAL (offset included), so results
	// from separate chunks share one index space.
	//
	// `cell_based`: when true the exit parameter is left unclamped (segment may run
	// past the target); when false it is clamped to the target endpoint (t <= 1).
	//
	// `rt1` is resized to the total intersection count for this chunk.
	void rayTrace1(const double3 *ll, const double3 *ur, size_t n_cells,
				   const double3 *ray_ipos, size_t i_offset, size_t n_ipos,
				   const double3 *ray_fpos, size_t f_offset, size_t n_fpos,
				   Raytrace1Result &rt1, bool cell_based);
}
