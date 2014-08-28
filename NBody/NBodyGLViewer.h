#pragma once

#include "..\AbstractResolver\AbstractResolver.h"

#include <GL\wglew.h>
#include <GL\freeglut.h>
#include <glm\glm.hpp>

#include <yaml-cpp\yaml.h>

#include <string>

struct ViewData{
	float zoom;
	float thetaX;
	float thetaY;
};

class NBodyGLViewer
{
private:

	//	Window Config
	std::string WINDOW_TITLE;
	int MAX_WIDTH;
	int MAX_HEIGHT;
	int MAX_FPS;
	float POINT_SIZE;
	float ZOOM;
	bool exportFrames;
	std::string exportLocation;
	std::string exportFilePrefix;

	//	Sim Config
	int NUM_PARTICLES;

	//	Various
	GLuint posArray;
	GLuint shaderProgramID;
	GLuint matrixID;
	glm::mat4 MVP;
	
	ViewData viewData;
	POINT mouseLoc;

	unsigned int elapsedTime;

	//	Buffer for saving frame pixels
	unsigned char *frameBuffer;
	int frameCounter;


	AbstractResolver *resolver;


	void initGL(int *argc, char **argv);

	GLuint loadShaders(const char *vertex_file_path, const char *fragment_file_path);

	friend void render();
	friend void windowResize(GLint newWidth, GLint newHeight);
	friend void mouseBtnDown(int button, int state, int x, int y);
	friend void mouseMove(int x, int y);
	friend void timer(int flag);
	void registerCallbacks();
	void exportFrame();

	NBodyGLViewer(){ }
	NBodyGLViewer(NBodyGLViewer &);
	void operator=(NBodyGLViewer &);



public:
	static NBodyGLViewer &getInstance(){
		static NBodyGLViewer instance;

		return instance;
	}

	bool init(int *argc, char **argv, AbstractResolver *resolver, YAML::Node &config);
	void start();

	~NBodyGLViewer(void);
};

