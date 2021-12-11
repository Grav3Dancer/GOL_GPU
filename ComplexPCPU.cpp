
#include "ComplexPCPU.h"

void iterationComplexParallel(unsigned char*& worldData, unsigned char* worldDataResult, size_t iterations, size_t worldHeight, size_t worldWidth, size_t blocksPerThread) {
	//if (worldWidth % 8 != 0) {
	//	std::cout << worldWidth % 8 << std::endl;
	//	std::cout << "complex pcpu error here: worldWidth % 8 != 0" << std::endl;
	//	return;
	//}

	size_t worldDataWidth = worldWidth;
	size_t worldSize = worldDataWidth * worldHeight;


	if (worldDataWidth % blocksPerThread != 0) {
		std::cout << worldDataWidth << std::endl;
		std::cout << blocksPerThread << std::endl;
		std::cout << worldDataWidth % blocksPerThread << std::endl;
		std::cout << "complex pcpu error here: worldDataWidth % blocksPerThread != 0" << std::endl;
		return;
	}

	//iterate(iterationsNumber, blocksPerThread, worldDataWidth, worldSize);

	std::cout << "start" << std::endl;

	auto evaluateCell = [&](size_t index) {
		index *= blocksPerThread;
		size_t x = (index + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
		size_t yAbs = (index / worldDataWidth) * worldDataWidth;
		size_t yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
		size_t yAbsDown = (yAbs + worldDataWidth) % worldSize;

		// Initialize data with previous byte and current byte.
		unsigned int data0 = (unsigned int)worldData[x + yAbsUp] << 16;
		unsigned int data1 = (unsigned int)worldData[x + yAbs] << 16;
		unsigned int data2 = (unsigned int)worldData[x + yAbsDown] << 16;

		x = (x + 1) % worldDataWidth;

		data0 |= (unsigned int)worldData[x + yAbsUp] << 8;
		data1 |= (unsigned int)worldData[x + yAbs] << 8;
		data2 |= (unsigned int)worldData[x + yAbsDown] << 8;

		for (unsigned int i = 0; i < blocksPerThread; ++i) {
			size_t oldX = x;  // Old x is referring to current center cell.
			x = (x + 1) % worldDataWidth;
			data0 |= (unsigned int)worldData[x + yAbsUp];
			data1 |= (unsigned int)worldData[x + yAbs];
			data2 |= (unsigned int)worldData[x + yAbsDown];

			unsigned int result = 0;
			for (unsigned int j = 0; j < 8; ++j) {
				// 23 ops.
				//unsigned int aliveCells = ((data0 >> 14) & 0x1u) + ((data0 >> 15) & 0x1u) + ((data0 >> 16) & 0x1u)
				//	+ ((data1 >> 14) & 0x1) + ((data1 >> 16) & 0x1)  // Do not count middle cell.
				//	+ ((data2 >> 14) & 0x1u) + ((data2 >> 15) & 0x1u) + ((data2 >> 16) & 0x1u);

				// 10 ops + modulo.
				//unsigned long long state = unsigned long long(((data0 & 0x1C000) >> 8)
				// | ((data1 & 0x14000) >> 11) | ((data2 & 0x1C000) >> 14));
				//assert(sizeof(state) == 8);
				//unsigned int aliveCells = unsigned int((state * 0x200040008001ULL & 0x111111111111111ULL) % 0xf);

				// 15 ops
				unsigned int aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
				aliveCells >>= 14;
				aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u)
					+ ((data2 >> 15) & 0x1u);

				result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);

				data0 <<= 1;
				data1 <<= 1;
				data2 <<= 1;
			}

			worldDataResult[oldX + yAbs] = (unsigned char)(result);
		}
	};

	for (size_t i = 0; i < iterations; ++i) {
		Concurrency::parallel_for<size_t>(0, worldSize / blocksPerThread, 1, evaluateCell);
		std::swap(worldData, worldDataResult);
	}

}

//bool iterate(size_t iterations, size_t blocksPerThread, size_t worldDataWidth, size_t worldSize) {
//
//	auto evaluateCell = [worldDataWidth, worldSize, blocksPerThread, this](size_t index) {
//		index *= blocksPerThread;
//		size_t x = (index + worldDataWidth - 1) % worldDataWidth;  // Start at block x - 1.
//		size_t yAbs = (index / worldDataWidth) * worldDataWidth;
//		size_t yAbsUp = (yAbs + worldSize - worldDataWidth) % worldSize;
//		size_t yAbsDown = (yAbs + worldDataWidth) % worldSize;
//
//		// Initialize data with previous byte and current byte.
//		unsigned int data0 = (unsigned int)m_bpcData[x + yAbsUp] << 16;
//		unsigned int data1 = (unsigned int)m_bpcData[x + yAbs] << 16;
//		unsigned int data2 = (unsigned int)m_bpcData[x + yAbsDown] << 16;
//
//		x = (x + 1) % worldDataWidth;
//
//		data0 |= (unsigned int)m_bpcData[x + yAbsUp] << 8;
//		data1 |= (unsigned int)m_bpcData[x + yAbs] << 8;
//		data2 |= (unsigned int)m_bpcData[x + yAbsDown] << 8;
//
//		for (unsigned int i = 0; i < blocksPerThread; ++i) {
//			size_t oldX = x;  // Old x is referring to current center cell.
//			x = (x + 1) % worldDataWidth;
//			data0 |= (unsigned int)m_bpcData[x + yAbsUp];
//			data1 |= (unsigned int)m_bpcData[x + yAbs];
//			data2 |= (unsigned int)m_bpcData[x + yAbsDown];
//
//			unsigned int result = 0;
//			for (unsigned int j = 0; j < 8; ++j) {
//				// 23 ops.
//				//unsigned int aliveCells = ((data0 >> 14) & 0x1u) + ((data0 >> 15) & 0x1u) + ((data0 >> 16) & 0x1u)
//				//	+ ((data1 >> 14) & 0x1) + ((data1 >> 16) & 0x1)  // Do not count middle cell.
//				//	+ ((data2 >> 14) & 0x1u) + ((data2 >> 15) & 0x1u) + ((data2 >> 16) & 0x1u);
//
//				// 10 ops + modulo.
//				//unsigned long long state = unsigned long long(((data0 & 0x1C000) >> 8)
//				// | ((data1 & 0x14000) >> 11) | ((data2 & 0x1C000) >> 14));
//				//assert(sizeof(state) == 8);
//				//unsigned int aliveCells = unsigned int((state * 0x200040008001ULL & 0x111111111111111ULL) % 0xf);
//
//				// 15 ops
//				unsigned int aliveCells = (data0 & 0x14000) + (data1 & 0x14000) + (data2 & 0x14000);
//				aliveCells >>= 14;
//				aliveCells = (aliveCells & 0x3) + (aliveCells >> 2) + ((data0 >> 15) & 0x1u)
//					+ ((data2 >> 15) & 0x1u);
//
//				result = result << 1 | (aliveCells == 3 || (aliveCells == 2 && (data1 & 0x8000u)) ? 1u : 0u);
//
//				data0 <<= 1;
//				data1 <<= 1;
//				data2 <<= 1;
//			}
//
//			m_bpcResultData[oldX + yAbs] = ubyte(result);
//		}
//	};
//
//	for (size_t i = 0; i < iterations; ++i) {
//		Concurrency::parallel_for<size_t>(0, worldSize / blocksPerThread, 1, evaluateCell);
//		std::swap(m_bpcData, m_bpcResultData);
//	}
//
//	return true;
//}