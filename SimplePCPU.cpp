#include "SimplePCPU.h"

ubyte getAliveNeighbours(bool* worldData, size_t x0, size_t x1, size_t x2,
                       size_t y0, size_t y1, size_t y2) {
  return worldData[x0 + y0] + worldData[x1 + y0] + worldData[x2 + y0] +
         worldData[x0 + y1] + worldData[x2 + y1] + worldData[x0 + y2] +
         worldData[x1 + y2] + worldData[x2 + y2];
}

void iterationSimpleParallel(bool*& worldData, bool* worldDataResult,
                             size_t worldDataLength, size_t iterationsNumber,
                             size_t worldHeight, size_t worldWidth) {
  for (size_t i = 0; i < iterationsNumber; i++) {
    auto evaluateCell = [&](size_t index) {
      size_t x1 = index % worldWidth;
      size_t y1 = index - x1;
      size_t y0 = (y1 + worldDataLength - worldWidth) % worldDataLength;
      size_t y2 = (y1 + worldWidth) % worldDataLength;
      size_t x0 = (x1 + worldWidth - 1) % worldWidth;
      size_t x2 = (x1 + 1) % worldWidth;

      ubyte aliveNeighbors = getAliveNeighbours(worldData, x0, x1, x2, y0, y1, y2);
      worldDataResult[y1 + x1] = aliveNeighbors == 3 || (aliveNeighbors == 2 && worldData[x1 + y1]) ? 1 : 0;
    };

    Concurrency::parallel_for<size_t>(0, worldWidth * worldHeight, 1, evaluateCell);
    std::swap(worldData, worldDataResult);
  }
}