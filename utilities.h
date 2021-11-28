#pragma once


#include <string>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdio>
#include <iostream>

#define THRESHOLD 0.3

void generateMap(bool*, size_t, size_t);
void generateMap(unsigned char*, size_t, size_t);
void prettyPrint(bool* map, int width, int height);
void prettyPrint(unsigned char* map, int width, int height);
bool compareMap(bool* map1, bool* map2, size_t width, size_t height);
unsigned int aliveCells(bool* map, size_t width, size_t height);
