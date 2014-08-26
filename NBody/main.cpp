#include "CudaStaticResolver.h"
#include "NBodyGLViewer.h"

#include <yaml-cpp\yaml.h>
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "libyaml-cppmdd.lib")
#else
#pragma comment(lib, "libyaml-cppmd.lib")
#endif



using namespace std;

int main(int argc, char **argv){

	if(argc < 2){
		cout<<"Usage: nbody.exe config_file"<<endl;
		system("PAUSE");
		return 1;
	}

	try{
		YAML::Node root = YAML::LoadFile(argv[1]);

		string module = root["Resolver"].as<string>();
		cout<<"Using module "<<module<<".dll"<<endl<<endl;

		NBodyGLViewer &viewer = NBodyGLViewer::getInstance();
		CudaStaticResolver *resolver = new CudaStaticResolver();

		YAML::Node &config = root["NBodyConfig"];
		if(!viewer.init(&argc, argv, resolver, config)){
			cout<<"Initialization failure"<<endl;
			system("PAUSE");
			return 1;
		}

		viewer.start();

	}catch(YAML::Exception &e){
		cout<<e.what()<<endl;
		system("PAUSE");
		return 1;
	}


	system("PAUSE");
	return 1;
}
