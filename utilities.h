#pragma once


#include <string>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstdio>
#include <iostream>

#define THRESHOLD 0.3

void generateMap(bool*, size_t);
void generateMap(unsigned char*, size_t, size_t);
void prettyPrint(bool* map, int width, int height);
void prettyPrint(unsigned char* map, int width, int height);
