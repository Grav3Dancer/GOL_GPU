#include "serialCpu.h"


//void clearData() {
//	delete[] worldData;
//	delete[] worldDataResult;
//	worldData = nullptr;
//	worldDataResult = nullptr;
//}
//
//void allocateData() {
//	clearData();
//	worldDataLength = worldWidth * worldHeight;
//	worldData = new bool[worldDataLength];
//	worldDataResult = new bool[worldDataLength];
//}
//
//void resizeWorld(size_t newWorldWidth, size_t newWorldHeight) {
//	clearData();
//	worldWidth = newWorldWidth;
//	worldHeight = newWorldHeight;
//}

ubyte getAliveCellsNum(bool* worldData, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2) {
	return worldData[x0 + y0] + worldData[x1 + y0] + worldData[x2 + y0]
		+ worldData[x0 + y1] + worldData[x2 + y1]
		+ worldData[x0 + y2] + worldData[x1 + y2] + worldData[x2 + y2];
}

void iterationSerial(bool* worldData, bool* worldDataResult, size_t iterationsNumber, size_t worldHeight, size_t worldWidth) {
	for (size_t i = 0; i < iterationsNumber; i++) {
		for (size_t y = 0; y < worldHeight; y++) {
			size_t y0 = ((y - 1 + worldHeight) % worldHeight) * worldWidth;
			size_t y1 = y * worldWidth;
			size_t y2 = ((y + 1) % worldHeight) * worldWidth;

			for (size_t x = 0; x < worldWidth; x++) {
				size_t x0 = (x - 1 + worldWidth) % worldWidth;
				size_t x1 = x;
				size_t x2 = (x + 1) % worldWidth;

				ubyte aliveNeighbors = getAliveCellsNum(worldData, x0, x1, x2, y0, y1, y2);
				if (aliveNeighbors == 3 || (aliveNeighbors == 2 && worldData[x1 + y1] == 1)) worldDataResult[x1 + y1] = 1;
				else worldDataResult[x1 + y1] = 0;
			}
		}
		std::swap(worldData, worldDataResult);
		std::cout << std::endl << "iteration "<< i << std::endl;
	}
}