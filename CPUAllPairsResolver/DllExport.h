#pragma once

#include "CPUAllPairsResolver.h"

#ifdef __dll__
#define DLL __declspec(dllexport)
#else
#define DLL __declspec(dllimport)
#endif 	// __dll__

extern "C" DLL void *buildResolver();