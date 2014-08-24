#include "CudaStaticResolver.cuh"
#include "NBodyGLViewer.h"



int main(int argc, char **argv){

	
	NBodyGLViewer &viewer = NBodyGLViewer::getInstance();

	CudaStaticResolver *resolver = new CudaStaticResolver();

	viewer.init(&argc, argv, resolver);
	viewer.start();


	return 1;
}
