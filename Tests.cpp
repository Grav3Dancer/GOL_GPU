#include "Tests.h"

void testGpuSimple(std::vector<int> iterations, std::vector<int> sizes, std::vector<int> threads, int repeat) {
	for (int thread: threads) {
		for (int size : sizes) {
			for (int noIterations : iterations) {
				size_t duration = 0;
				for (int i = 0; i < repeat; i++) {
					bool* map = new bool[size * size];
					bool* mapBuffer = new bool[size * size];
					generateMap(map, size, size);
					auto start = std::chrono::high_resolution_clock::now();
					runEvaluateSimple(map, mapBuffer, size, size, noIterations, thread);
					auto end = std::chrono::high_resolution_clock::now();
					duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				}

				std::cout << size << ";" << noIterations << ";"  <<duration/repeat << ";" << thread << ";" << std::endl;
			}
		}
	}
}


void testGpuAdvanced(std::vector<int> iterations, std::vector<int> sizes, std::vector<int> threads, std::vector<int> blockSizes, int repeat) {
	for (int blockSize : blockSizes) {
		for (int thread : threads) {
			for (int size : sizes) {
				for (int noIterations : iterations) {
					size_t duration = 0;
					for (int i = 0; i < repeat; i++) {
						unsigned char* map = new unsigned char[size * size/8];
						unsigned char* mapBuffer = new unsigned char[size * size/8];
						generateMap(map, size/8, size);
						auto start = std::chrono::high_resolution_clock::now();
						runEvaluateAdvanced(map, mapBuffer, size/8, size, noIterations, blockSize ,thread);
						auto end = std::chrono::high_resolution_clock::now();
						duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
					}

					std::cout << size << ";" << noIterations << ";" << duration / repeat << ";" << thread << ";" << blockSize << std::endl;
				}
			}
		}
	}
}

void testCpuSimple(std::vector<int> iterations, std::vector<int> sizes, int repeat) {
	for (int size : sizes) {
		for (int noIterations : iterations) {
			size_t duration = 0;
			for (int i = 0; i < repeat; i++) {
				bool* map = new bool[size * size];
				bool* mapBuffer = new bool[size * size];
				generateMap(map, size, size);
				auto start = std::chrono::high_resolution_clock::now();
				iterationSerial(map, mapBuffer, noIterations, size, size);
				auto end = std::chrono::high_resolution_clock::now();
				duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			}

			std::cout << size << ";" << noIterations << ";" << duration / repeat << ";" << std::endl;
		}
	}
}

void testPCpuSimple(std::vector<int> iterations, std::vector<int> sizes, int repeat) {
	for (int size : sizes) {
		for (int noIterations : iterations) {
			size_t duration = 0;
			for (int i = 0; i < repeat; i++) {
				bool* map = new bool[size * size];
				bool* mapBuffer = new bool[size * size];
				generateMap(map, size, size);
				auto start = std::chrono::high_resolution_clock::now();
				iterationSimpleParallel(map, mapBuffer, size*size,noIterations, size, size);
				auto end = std::chrono::high_resolution_clock::now();
				duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			}

			std::cout << size << ";" << noIterations << ";" << duration / repeat << ";" << std::endl;
		}
	}
}

void testPCpuAdvanced(std::vector<int> iterations, std::vector<int> sizes, std::vector<int> blockSizes, int repeat) {
	for (int blockSize : blockSizes) {
		for (int size : sizes) {
			for (int noIterations : iterations) {
				size_t duration = 0;
				for (int i = 0; i < repeat; i++) {
					unsigned char* map = new unsigned char[size * size / 8];
					unsigned char* mapBuffer = new unsigned char[size * size / 8];
					generateMap(map, size / 8, size);
					auto start = std::chrono::high_resolution_clock::now();
					iterationComplexParallel(map, mapBuffer, noIterations, size, size / 8, blockSize);
					auto end = std::chrono::high_resolution_clock::now();
					duration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				}

				std::cout << size << ";" << noIterations << ";" << duration / repeat << ";" << std::endl;
			}
		}
	}
}


void testAll() {
	std::vector<int> iterations = { 100, 200, 500};
	std::vector<int> sizes = { 128, 256, 512};
	std::vector<int> threads = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
	std::vector<int> cellSizes = { 1,2,3,4,5,6,7,8 };


	std::cout << "GPU Simple" << std::endl;
	std::cout << "Map size; No iterations; time [ms];Threads; " << std::endl;
	testGpuSimple(iterations, sizes, threads, 1);

	std::cout << "GPU Advanced" << std::endl;
	std::cout << "Map size; No iterations; time [ms];Threads;cellSize;" << std::endl;
	testGpuAdvanced(iterations, sizes, threads,cellSizes, 1);



	std::cout << "PCPU Simple" << std::endl;
	std::cout << "Map size; No iterations; time [ms];" << std::endl;
	testPCpuSimple(iterations, sizes, 1);

	std::cout << "PCPU Advanced" << std::endl;
	std::cout << "Map size; No iterations; time [ms];cellSize;" << std::endl;
	testPCpuAdvanced(iterations, sizes, cellSizes, 1);



	std::cout << "CPU Simple" << std::endl;
	std::cout << "Map size; No iterations; time [ms];" << std::endl;
	testCpuSimple(iterations, sizes, 1);

}