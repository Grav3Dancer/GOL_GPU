#include "utilities.h"

unsigned char _generateRandomCellGroup();
unsigned int _countSetBits(unsigned char x);
unsigned char boolToChar(bool*);

void generateMap(bool* map, size_t width, size_t height, bool printInfo) {
	srand(time(NULL));

	unsigned int counter = 0;

	for (size_t i = 0;i < width; i++) {
		for (size_t j = 0;j < height; j++) {
			map[j * width + i] = ((float)rand() / (float)RAND_MAX) < THRESHOLD;
			counter += map[j * width + i];
		}
	}

	if (printInfo) {

	std::cout << "Number of non zero elements: " << counter << std::endl;
	std::cout << "Percent: " << (float)counter / (float)(width * height) << std::endl;
	}
}

void generateMap(unsigned char* map, size_t width, size_t height, bool printInfo) {
	srand(time(NULL));

	unsigned int counter = 0;

	for (size_t i = 0; i < width * height; i++) {
		map[i] = _generateRandomCellGroup();
	}

	if (printInfo) {
	std::cout << "Number of non zero elements: " << counter << std::endl;
	std::cout << "Percent: " << (float)counter / (float)(width * height) << std::endl;
	}
}

void copyBoolToCharMap(bool* map1, unsigned char* map2, size_t width, size_t height) {
	//for (size_t i = 0;i < width; i++) {
	//	for (size_t j = 0;j < height; j++) {
	//		unsigned char val = 0;
	//		std::cout << j * width + i << ": ";
	//		for (size_t k = 0; k < 8;k++) {
	//			std::cout << j * width + i * 8 + k << " ";
	//			val <<= 1;
	//			if (map1[j*width+i*8+k]) val |= 1;
	//		}
	//		std::cout << std::endl;
	//		std::cout << "Copied " << (int)val << " to " << i << " " << j << std::endl;
	//		map2[j * width + i] = val;
	//	}
	//}
	for (size_t i = 0; i < width * height; i++) {
		unsigned char val = 0;
		for (size_t k = 0; k < 8;k++) {
			val <<= 1;
			if (map1[i * 8 + k]) val |= 1;
		}
		map2[i] = val;
	}
}


bool compareMap(bool* map1, bool* map2, size_t width, size_t height) {
	for (size_t i = 0; i < width; i++) {
		for (size_t j = 0;j < height;j++) {
			if (map1[j * width + i] != map2[j * width + i])
				return false;
		}
	}
	return true;
}

bool compareBoolToCharMap(bool* map1, unsigned char* map2, size_t width, size_t height) {
	//for (size_t i= 0; i < width; i++) {
	//	for (size_t j = 0;j < height;j++) {
	//		unsigned char val = 0;
	//		for (size_t k = 0; k < 8;k++) {
	//			val <<= 1;
	//			if (map1[j * width + i * 8 + k]) val |= 1;
	//		}
	//		if (map2[j * width + i] != val) { 
	//			std::cout << "Difference found at " << j*width << " " << i << std::endl;
	//			std::cout << (int)map2[j * width + i] << " ";
	//			for (size_t k = 0; k < 8;k++) {
	//				std::cout << (int)map1[j * width + i * 8 + k];
	//			}
	//			std::cout << std::endl;
	//			return false; 
	//		}
	//	}
	//}
	//return true;
	for (size_t i = 0; i < width * height; i++) {
		unsigned char val = 0;
		for (size_t k = 0; k < 8;k++) {
			val <<= 1;
			if (map1[i * 8 + k]) val |= 1;
		}
		if (map2[i] != val) return false;
	}
	return true;
}


unsigned int aliveCells(bool* map, size_t width, size_t height) {
	unsigned int counter = 0;
	for (size_t i = 0; i < width; i++) {
		for (size_t j = 0;j < height;j++) {
			if (map[j * width + i])
				counter++;
		}
	}
	return counter;
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
	//for (size_t i = 0; i < width; i++) {
	//	for (size_t j = 0; j < height; j++) {
	//		std::cout << map[j * width + i] << " ";
	//	}
	//	std::cout << std::endl;
	//}
	for (size_t i = 0; i < width * height;i++) {
		if (i % 8 == 0) {
			std::cout << " ";
		}
		std::cout << map[i];
	}
}

void prettyPrint(unsigned char* map, int width, int height) {
	//for (size_t i = 0; i < width; i++) {
	//	for (size_t j = 0; j < height; j++) {
	//		std::cout << (int)map[i * width + j] << " ";
	//	}
	//	std::cout << std::endl;
	//}
	for (size_t i = 0; i < width * height;i++) {
		std::cout << (int)map[i] << " ";
	}
}

unsigned char boolToChar(bool* b) {
	unsigned char val = 0;
	for (int i = 0; i < 8; i++) {
		val <<= 1;
		if (b[i]) val |= 1;
	}
	return val;
}