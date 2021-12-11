#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void runEvaluateAdvanced(unsigned char*& map, unsigned char* mapBuffer, size_t mapWidth, size_t mapHeight, size_t iterations, size_t bytesPerThread, unsigned int threads);