#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <utility>
#include <iostream>

typedef unsigned char ubyte;

ubyte getAliveCellsNum(bool* data, size_t x0, size_t x1, size_t x2, size_t y0, size_t y1, size_t y2);
void iterationSerial(bool*& worldData, bool* worldDataResult, size_t iterationsNumber, size_t worldHeight, size_t worldWidth);
