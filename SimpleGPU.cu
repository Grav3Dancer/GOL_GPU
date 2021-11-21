#include "SimpleGPU.cuh"

__global__ void evaluateSimple(bool* map, unsigned int mapWidth, unsigned int mapHeight, bool* resultMap) {
	unsigned int worldSize = mapWidth * mapHeight;

	for (unsigned int cell = blockIdx.x * blockDim.x + threadIdx.x; cell < worldSize; cell += blockDim.x * gridDim.x) {
		unsigned int x = cell % mapWidth;
		unsigned int yAbs = cell - x;
		unsigned int xLeft = (x + mapWidth - 1) % mapWidth;
		unsigned int xRight = (x + 1) % mapWidth;
		unsigned int yAbsUp = (yAbs + worldSize - mapWidth) % worldSize;
		unsigned int yAbsDown = (yAbs + mapWidth) % worldSize;

		unsigned int aliveCells = 
			map[xLeft + yAbsUp] + map[x + yAbsUp] + map[xRight + yAbsUp]
			+ map[xLeft + yAbs] + map[xRight + yAbs]
			+ map[xLeft + yAbsDown] + map[x + yAbsDown] + map[xRight + yAbsDown];

		resultMap[x + yAbs] = aliveCells == 3 || (aliveCells == 2 && map[x + yAbs]) ? 1 : 0;
	}
}

void runEvaluateSimple(bool*& map, bool* mapBuffer, size_t mapWidth, size_t mapHeight, size_t iterations, unsigned int threads) {
	if ((mapWidth * mapHeight) % threads != 0) {
		std::cout << "Error - wrong number of threads" << std::endl;
		return;
	}

	size_t reqBlocks = (mapWidth * mapHeight) / threads;
	unsigned int blocks = std::min((size_t)32768, reqBlocks);

	for (size_t i = 0; i < iterations; i++) {
		evaluateSimple << <blocks, threads >> > (map, mapWidth, mapHeight, mapBuffer);
		std::swap(map, mapBuffer);
	}
}