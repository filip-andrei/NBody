#pragma once
#include <GL\glew.h>

void genBodies(GLuint posVBO, GLuint velVBO, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm);
void moveBodiesByDT_staticPotential(GLuint posVBO, GLuint velVBO, float dT, float bodyMass, int NUM_PARTICLES, float Ms, float Rs, float Mdm, float Rdm);
void moveBodiesByDT_NBody(GLuint posVBO, GLuint velVBO, float dT, float bodyMass, float Mdm, float Rdm, int NUM_PARTICLES);