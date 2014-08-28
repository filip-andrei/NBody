#include <cuda_runtime_api.h>

//	Kilometers per Parsec
__device__ float kmPerPc = 3.0857e13f;

//	Gravitational constant in ( pc / SM ) * (km/s)^2
__device__ float G = 4.302e-3f;

//	Conversion factor from km/s to pc/Myr
__device__ float velConvFactor = 1.0226f;

//	Number of seconds per Myr
__device__ float secPerMYr = 3.15569e13f;