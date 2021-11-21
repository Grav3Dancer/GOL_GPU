#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

void runEvaluateSimple(bool*& map, bool* mapBuffer, size_t mapWidth, size_t mapHeight, size_t iterations, unsigned int threads);