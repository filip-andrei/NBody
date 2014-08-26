#pragma once

#include <GL\glew.h>
#include <yaml-cpp\yaml.h>

class AbstractResolver
{
public:

	virtual void advanceTimeStep()=0;

	virtual void setPosBufferID(GLuint vboID)=0;

	virtual bool initialize(YAML::Node &config)=0;

	virtual ~AbstractResolver(void){ };
};

