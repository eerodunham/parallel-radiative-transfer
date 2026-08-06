#include "npy.hpp"
#include <types.hpp>
#include "raytracer.h"

int main(int argc, const char *argv)
{
	using namespace Raytracer;
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << "path/to/data << [number of ipos] [number of fpos]\n";
		return 1;
	}
}