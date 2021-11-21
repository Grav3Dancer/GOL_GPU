#include "utilities.h"

unsigned char _generateRandomCellGroup();
unsigned int _countSetBits(unsigned char x);

void generateMap(bool* map, size_t N) {
	srand(time(NULL));

	unsigned int counter = 0;

	for (size_t i = 0;i < N; i++) {
		for (size_t j = 0;j < N; j++) {
			map[j * N + i] = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
			counter += map[j * N + i];
		}
	}

	std::cout << "Number of non zero elements: " << counter << std::endl;
	std::cout << "Percent: " << (float)counter / (float)(N * N) << std::endl;
}

void generateMap(unsigned char* map, size_t width, size_t height) {
	srand(time(NULL));

	unsigned int counter = 0;

	for (size_t i = 0;i < width/8; i++) {
		for (size_t j = 0;j < height; j++) {
			map[j * height + i] = _generateRandomCellGroup();
			counter += _countSetBits(map[j * height + i]);
		}
	}

	std::cout << "Number of non zero elements: " << counter << std::endl;
	std::cout << "Percent: " << (float)counter / (float)(width * height) << std::endl;
}


unsigned char _generateRandomCellGroup() {
	unsigned char generated = 0;
	//for (size_t bit = 0; bit < 8;bit++) {
	//	generated += (((float)rand() / (float)RAND_MAX) < THRESHOLD) << bit;
	//}
	generated = (((float)rand() / (float)RAND_MAX) < THRESHOLD) | 
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 1 | 
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 2 | 
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 3 | 
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 4 |
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 5 | 
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 6 | 
		(((float)rand() / (float)RAND_MAX) < THRESHOLD) << 7;
	return generated;
}

unsigned int _countSetBits(unsigned char x) {
	unsigned int count = 0;
	while (x != 0) {
		if ((x & 0x1) == 1) count++;
		x = x >> 1;
	}
	return count;
}


void prettyPrint(bool* map, int width, int height) {
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			std::cout << map[i * height + j] << " ";
		}
		std::cout << std::endl;
	}
}

void prettyPrint(unsigned char* map, int width, int height) {
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width/8; j++) {
			std::cout << map[i * height + j] * 1 << " ";
		}
		std::cout << std::endl;
	}
}