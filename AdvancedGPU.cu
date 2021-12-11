#include "SimpleGPU.cuh"
#include "utilities.h"

__global__ void evaluateAdvanced(unsigned char* map, unsigned int mapWidth, unsigned int mapHeight, unsigned int bytesPerThread, unsigned char* resultMap) {
	unsigned int worldSize = mapWidth * mapHeight;

	for (unsigned int cell = (blockIdx.x * blockDim.x + threadIdx.x) * bytesPerThread; cell < worldSize; cell += blockDim.x * gridDim.x * bytesPerThread) {
		unsigned int x = (cell + mapWidth - 1) % mapWidth; // x-1 block
		unsigned int yAbs = (cell/mapWidth) * mapWidth;
		unsigned int yAbsUp = (yAbs + worldSize - mapWidth) % worldSize;
		unsigned int yAbsDown = (yAbs + mapWidth) % worldSize;

		// Initialize data (prev byte and curr byte)
		unsigned int data0 = (unsigned int)map[x + yAbsUp] << 16;
		unsigned int data1 = (unsigned int)map[x + yAbs] << 16;
		unsigned int data2 = (unsigned int)map[x + yAbsDown] << 16;

		x = (x + 1) % mapWidth;
		data0 |= (unsigned int)map[x + yAbsUp] << 8;
		data1 |= (unsigned int)map[x + yAbs] << 8;
		data2 |= (unsigned int)map[x + yAbsDown] << 8;

		for (unsigned int i = 0; i < bytesPerThread; i++) {
			unsigned int oldX = x;
			x = (x + 1) % mapWidth;
			data0 |= (unsigned int)map[x + yAbsUp];
			data1 |= (unsigned int)map[x + yAbs];
			data2 |= (unsigned int)map[x + yAbsDown];

			unsigned int result = 0;
			for (unsigned int j = 0; j < 8;j++) {
				unsigned int aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
				aliveCells >>= 14;
				aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u)
					+ ((data2 >> 15) & 0x1u);

				result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

				data0 <<= 1;
				data1 <<= 1;
				data2 <<= 1;
			}

			resultMap[oldX + yAbs] = result;
		}

	}
}

void runEvaluateAdvanced(unsigned char*& map, unsigned char* mapBuffer, size_t mapWidth, size_t mapHeight, size_t iterations, size_t bytesPerThread, unsigned int threads) {
	size_t size = mapWidth * mapHeight;
	if (size % threads != 0) {
		std::cout << "Error - wrong number of threads" << std::endl;
		return;
	}

	unsigned char* devMap = nullptr;
	unsigned char* devMapBuffer = nullptr;
	// Allocate GPU buffers
	cudaMalloc((void**)&devMap, size);
	cudaMalloc((void**)&devMapBuffer, size);

	cudaMemcpy(devMap, map, size, cudaMemcpyHostToDevice);
	cudaMemcpy(devMapBuffer, map, size, cudaMemcpyHostToDevice);

	size_t reqBlocks = (mapWidth * mapHeight) / threads;
	unsigned int blocks = std::min((size_t)32768, reqBlocks);

	//std::cout << "Iteration 0" << std::endl;
	//prettyPrint(map, mapWidth, mapHeight);
	for (size_t i = 0; i < iterations; i++) {
		//std::cout << "Runing simple" << std::endl;
		evaluateAdvanced << <blocks, threads >> > (devMap, mapWidth, mapHeight, bytesPerThread, devMapBuffer);
		std::swap(devMap, devMapBuffer);
	}

	cudaMemcpy(map, devMap, size, cudaMemcpyDeviceToHost);

	// Clean up
	cudaFree(devMap);
	cudaFree(devMapBuffer);

	cudaDeviceSynchronize();
}