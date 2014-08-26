#define __dll__

#include <Windows.h>
#include "DllExport.h"

int WINAPI DllEntryPoint(HINSTANCE hinst, unsigned long reason, void*){
	return 1;
}

DLL void * buildResolver(){
	return static_cast<void *>(new CudaStaticResolver());
}