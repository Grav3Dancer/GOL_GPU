#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <iostream>
#include <ppl.h>


void iterationComplexParallel(unsigned char*& worldData, unsigned char* worldDataResult, size_t iterations, size_t worldHeight, size_t worldWidth, size_t blocksPerThread);
//bool iterate(size_t iterationsNumber, size_t blocksPerThread, size_t worldDataWidth, size_t worldSize);