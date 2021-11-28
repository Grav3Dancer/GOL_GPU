
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <string>
#include <conio.h>

#include <stdio.h>
#include "utilities.h"
#include "SimpleGPU.cuh"
#include "SimpleCPU.h"
#include "SimplePCPU.h"

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	int iterations = 10000;
	int threads = 256;
	int width = 480;
	int height = 480;
	int size = width * height;

	bool* map = new bool[size];
	bool* mapBuffer = new bool[size];

	generateMap(map, width, height);
	//std::cout << "Iteration 0" << std::endl;
	//prettyPrint(map, width, height);

	bool* mapGPU = new bool[size];
	std::copy(map, map+size, mapGPU);
	bool* mapCPU = new bool[size];
	std::copy(map, map + size, mapCPU);
	bool* mapPCPU = new bool[size];
	std::copy(map, map + size, mapPCPU);

	auto start = std::chrono::high_resolution_clock::now();
	runEvaluateSimple(mapGPU, mapBuffer, width, height, iterations, threads);

	auto startCPU = std::chrono::high_resolution_clock::now();
	//iterationSerial(mapCPU, mapBuffer, iterations, height, width);

	auto startPCPU = std::chrono::high_resolution_clock::now();
	//iterationSimpleParallel(mapPCPU, mapBuffer, size, iterations, height, width);

	auto stop = std::chrono::high_resolution_clock::now();

	std::cout << std::endl;

	std::cout << std::endl;
	auto durationGPU = std::chrono::duration_cast<std::chrono::milliseconds>(startCPU - start);
	auto durationCPU = std::chrono::duration_cast<std::chrono::milliseconds>(startPCPU - startCPU);
	auto durationPCPU = std::chrono::duration_cast<std::chrono::milliseconds>(stop - startPCPU);
	std::cout << "gpu: " << durationGPU.count() << std::endl;
	std::cout << "cpu: " << durationCPU.count() << std::endl;
	std::cout << "pcpu: " << durationPCPU.count() << std::endl;

	if (!compareMap(mapCPU, mapGPU, width, height)) {
		std::cout << "GPU incorrect result map" << std::endl;
	}
	if (!compareMap(mapCPU, mapPCPU, width, height)) {
		std::cout << "PCPU incorrect result map" << std::endl;
	}

	std::cout << "Alive cells: " << aliveCells(mapGPU, width, height) << " " << aliveCells(mapCPU, width, height) << " " << aliveCells(mapPCPU, width, height) << std::endl;

	//prettyPrint(map, width, height);

	std::cout << "press any key to exit";
	getch();

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
	int* dev_a = 0;
	int* dev_b = 0;
	int* dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
