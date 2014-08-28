#include <GL\glew.h>

void genBodies(float *d_pos, float *d_vel, float *d_mas, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, int blockSize);
void moveBodiesByDT(float *d_pos, float *d_vel, float *d_mas, float dT, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm, int blockSize);