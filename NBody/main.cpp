//	OpenGL lib

#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>

// GLM
#include <glm\glm.hpp>
#include <glm\gtx\transform.hpp>

//	Cuda stuff
#include <cuda.h>
#include <cuda_gl_interop.h>

//	Various
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "shader.h"
#include "cuda_kernel.cuh"

//	Perspective stuff
struct ViewData{
	float zoom;
	float thetaX;
	float thetaY;
} viewData;


//	Window Config
const int MAX_WIDTH = 800;
const int MAX_HEIGHT = 800;
const int MAX_FPS = 60;
const float POINT_SIZE = 1.0f;
const float ZOOM = 15.0f;

//	Sim Config
const int NUM_PARTICLES = 3000000;
const float Rs = 3130.0f;					//	Scale radius for stellar density (Pcs)
const float Mt = 100.0f;					//	Total mass of galaxy (in solar masses)


//	Various
GLuint posArray;		//	Array in gpu memory containing (x,y,z) positional information	(Pcs)
GLuint velArray;		//	Array in gpu memory containing (x,y,z) velocity information		(Km/s)
GLuint shaderProgramID;
GLuint matrixID;
glm::mat4 MVP;
POINT mouseLoc;


void initOpenGL(int *argc, char **argv){
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(MAX_WIDTH, MAX_HEIGHT);
	glutCreateWindow("N-body window");

	glewInit();
}

void init(){

	viewData.thetaX = 0.0f;
	viewData.thetaY = 0.0f;
	viewData.zoom = 1.0f / ZOOM;
	mouseLoc.x = 0;
	mouseLoc.y = 0;


	shaderProgramID = LoadShaders("vertexShader.glsl", "fragmentShader.glsl");
	

	glGenBuffers(1, &posArray);
	glBindBuffer(GL_ARRAY_BUFFER, posArray);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * NUM_PARTICLES, 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGLRegisterBufferObject(posArray);

	float *d_posArray;

	cudaGLMapBufferObject( (void **)&d_posArray, posArray);

	genBodies(d_posArray, Rs, NUM_PARTICLES);

	cudaGLUnmapBufferObject(posArray);
	
	glm::mat4 model = glm::scale(glm::vec3(1.0));
	glm::mat4 view = glm::lookAt(glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,1.0,0.0));
	glm::mat4 projection = glm::perspective(45.0f, (float)MAX_WIDTH/MAX_HEIGHT, 0.1f, 100.0f);

	MVP = projection * view * model;

	matrixID = glGetUniformLocation(shaderProgramID, "MVP");
}



void renderScene(){

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPointSize(POINT_SIZE);

	glm::mat4 model = glm::scale(glm::vec3(viewData.zoom));
	model *= glm::rotate(model, viewData.thetaY, glm::vec3(1,0,0));
	model *= glm::rotate(model, viewData.thetaX, glm::vec3(0,0,1));

	glm::mat4 view = glm::lookAt(glm::vec3(0.0,0.0,1.0), glm::vec3(0.0,0.0,0.0), glm::vec3(0.0,1.0,0.0));

	glm::mat4 projection = glm::perspective(45.0f, (float)MAX_WIDTH/MAX_HEIGHT, 0.1f, 100.0f);

	MVP = projection * view * model;

	glUseProgram(shaderProgramID);
	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, posArray);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		0
	);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

	glDisableVertexAttribArray(0);

	glutSwapBuffers();
}

void timer(int flag) {
	int drawStartTime = clock();
    glutPostRedisplay();
	int drawEndTime = clock();

	float delayToNextFrame =  (CLOCKS_PER_SEC/MAX_FPS) - (drawEndTime-drawStartTime);
    delayToNextFrame = floor(delayToNextFrame+0.5);
    delayToNextFrame < 0 ? delayToNextFrame = 0 : NULL;

    glutTimerFunc(delayToNextFrame, timer, 0);
}

void mouseBtnDown(int button, int state, int x, int y){
	if(state == GLUT_DOWN){
		mouseLoc.x = x;
		mouseLoc.y = y;
	}

	if(button == 3){
		viewData.zoom += 0.05f * viewData.zoom;
	}
	else if(button == 4){
		viewData.zoom -= 0.05f * viewData.zoom;
	}
}

void mouseMove(int x, int y){
	viewData.thetaX += (x - mouseLoc.x) * 0.5f;
	viewData.thetaY += (y - mouseLoc.y) * 0.5f;


	mouseLoc.x = x;
	mouseLoc.y = y;
}

int main(int argc, char **argv){
	
	//	Initialize GL subsystem
	initOpenGL(&argc, argv);
	init();

	//	Register callback functions
	glutDisplayFunc(renderScene);
	glutMouseFunc(mouseBtnDown);
	glutMotionFunc(mouseMove);
	glutTimerFunc(30, timer, 0);
	glutMainLoop();


	return 1;
}