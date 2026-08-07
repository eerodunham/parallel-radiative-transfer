// Validation test for the CUDA ray_trace_1 against the numpy oracle.
//
// Loads synthetic inputs through the Halo interface, runs the chunked rayTrace1, then
// compares the accumulated (target, cell, star) -> dr intersections against the
// reference produced by gen_rt1_reference.py. Because the fill order differs between
// the two (per-ray vs per-cell), results are matched by the (target, cell, star) key
// rather than positionally.
//
// argv: <data_path> [halo_name]   (defaults: "data" "synth")
// exit code 0 = pass, 1 = validation mismatch, 2 = setup/IO error.

#include "halo.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <unordered_map>

using namespace Raytracer;

int main(int argc, char **argv)
{
	const std::string data_path = (argc > 1) ? argv[1] : "data";
	const std::string halo_name = (argc > 2) ? argv[2] : "synth";

	// Small chunk caps so the multi-chunk host loop is actually exercised.
	Halo halo(/*cell_based=*/true, /*max_i=*/7, /*max_f=*/100);
	try
	{
		halo.init(data_path, halo_name);
	}
	catch (const std::exception &e)
	{
		std::fprintf(stderr, "Halo::init failed: %s\n", e.what());
		return 2;
	}

	const Rt1HostResult &res = halo.rayTrace1();

	npy::npy_data<double> ref_dr;
	npy::npy_data<uint32_t> ref_tg, ref_cl, ref_st;
	try
	{
		const std::string dir = data_path + "/" + halo_name + "/";
		ref_dr = npy::read_npy<double>(dir + "ref_dr.npy");
		ref_tg = npy::read_npy<uint32_t>(dir + "ref_target.npy");
		ref_cl = npy::read_npy<uint32_t>(dir + "ref_cell.npy");
		ref_st = npy::read_npy<uint32_t>(dir + "ref_star.npy");
	}
	catch (const std::exception &e)
	{
		std::fprintf(stderr, "failed to read reference .npy: %s\n", e.what());
		return 2;
	}

	const size_t nref = ref_dr.data.size();
	const size_t ncuda = res.size();
	std::printf("intersections: cuda=%zu  reference=%zu\n", ncuda, nref);

	// Pack (target, cell, star) into one 64-bit key. Each triple is unique per intersection.
	const uint64_t n_cells = halo.numCells();
	const uint64_t n_stars = halo.numStars();
	auto make_key = [&](uint64_t t, uint64_t c, uint64_t s) -> uint64_t
	{
		return (t * n_cells + c) * n_stars + s;
	};

	std::unordered_map<uint64_t, double> refmap;
	refmap.reserve(nref * 2);
	for (size_t i = 0; i < nref; ++i)
		refmap[make_key(ref_tg.data[i], ref_cl.data[i], ref_st.data[i])] = ref_dr.data[i];

	const double rel_tol = 1e-9;
	const double abs_tol = 1e-3; // cm; only guards genuinely near-zero spans
	size_t missing = 0;			 // cuda intersection with no matching reference key
	size_t mismatches = 0;		 // matched key but dr disagrees
	double max_rel = 0.0;
	for (size_t i = 0; i < ncuda; ++i)
	{
		const uint64_t k = make_key(res.target_ind[i], res.cell_ind[i], res.star_ind[i]);
		auto it = refmap.find(k);
		if (it == refmap.end())
		{
			++missing;
			continue;
		}
		const double a = res.dr[i], b = it->second;
		const double denom = std::max(std::fabs(b), 1.0);
		const double rel = std::fabs(a - b) / denom;
		max_rel = std::max(max_rel, rel);
		if (std::fabs(a - b) > abs_tol && rel > rel_tol)
			++mismatches;
		refmap.erase(it);
	}
	const size_t ref_only = refmap.size(); // reference keys the CUDA pass never produced

	std::printf("count_match=%d  missing_in_ref=%zu  dr_mismatches=%zu  ref_only=%zu  max_rel=%.3e\n",
				(int)(ncuda == nref), missing, mismatches, ref_only, max_rel);

	const bool ok = (ncuda == nref) && missing == 0 && mismatches == 0 && ref_only == 0;
	std::printf("%s\n", ok ? "RT1 VALIDATION PASSED" : "RT1 VALIDATION FAILED");
	return ok ? 0 : 1;
}
