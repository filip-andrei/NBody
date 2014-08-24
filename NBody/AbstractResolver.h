#pragma once

#include <GL\glew.h>

class AbstractResolver
{
public:

	virtual void advanceTimeStep()=0;

	virtual void loadSimConfig()=0;
	virtual void setPosBufferID(GLuint vboID)=0;

	virtual void initialize()=0;

	virtual ~AbstractResolver(void){ };
};

