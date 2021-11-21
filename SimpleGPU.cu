#include "SimpleGPU.cuh"
#include "utilities.h"

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

		resultMap[x + yAbs] = (aliveCells == 3 || (aliveCells == 2 && map[x + yAbs]) ? 1 : 0);
	}
}

void runEvaluateSimple(bool*& map, bool* mapBuffer, size_t mapWidth, size_t mapHeight, size_t iterations, unsigned int threads) {
	size_t size = mapWidth * mapHeight;
	if (size % threads != 0) {
		std::cout << "Error - wrong number of threads" << std::endl;
		return;
	}

	bool* devMap = nullptr;
	bool* devMapBuffer = nullptr;
	// Allocate GPU buffers
	cudaMalloc((void**)&devMap, size);
	cudaMalloc((void**)&devMapBuffer, size);

	cudaMemcpy(devMap, map, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devMapBuffer, map, size, cudaMemcpyHostToDevice);

	size_t reqBlocks = (mapWidth * mapHeight) / threads;
	unsigned int blocks = std::min((size_t)32768, reqBlocks);

	std::cout << "Iteration 0" << std::endl;
	prettyPrint(map, mapWidth, mapHeight);
	for (size_t i = 0; i < iterations; i++) {
		std::cout << "Runing simple" << std::endl;
		evaluateSimple << <blocks, threads >> > (devMap, mapWidth, mapHeight, devMapBuffer);
		cudaDeviceSynchronize();
		std::swap(devMap, devMapBuffer);
	}

	cudaMemcpy(map, devMap, size, cudaMemcpyDeviceToHost);
	std::cout << "Iteration 1" << std::endl;
	prettyPrint(map, mapWidth, mapHeight);
}