#include "..\AbstractResolver\AbstractResolver.h"
#include "NBodyGLViewer.h"

#include <yaml-cpp\yaml.h>
#include <iostream>


#ifdef _DEBUG
#pragma comment(lib, "libyaml-cppmdd.lib")
#else
#pragma comment(lib, "libyaml-cppmd.lib")
#endif



using namespace std;

typedef AbstractResolver *(CALLBACK *buildResolverFunc)();

int main(int argc, char **argv){

	if(argc < 2){
		cout<<"Usage: nbody.exe config_file"<<endl;
		system("PAUSE");
		return 1;
	}

	try{
		YAML::Node root = YAML::LoadFile(argv[1]);

		string moduleName;
		if(root["Resolver"]){
			moduleName = root["Resolver"].as<string>();
		}
		else{
			cout<<"No Resolver specified in config"<<endl;
			system("PAUSE");
			return 1;
		}

		char modulePath[1024];
		sprintf_s(modulePath, 1024, "%s.dll", moduleName.c_str());

		cout<<"Using module "<<modulePath<<endl<<endl;

		HINSTANCE module = LoadLibrary(modulePath);

		if(!module){
			cout<<"Could not find module "<<modulePath<<endl;
			system("PAUSE");
			return 1;
		}

		buildResolverFunc buildResolver = (buildResolverFunc)GetProcAddress(module, "buildResolver");

		if(!buildResolver){
			cout<<"Could not get buildResolver() from module"<<endl;
			system("PAUSE");
			return 1;
		}

		
		AbstractResolver *resolver = buildResolver();

		NBodyGLViewer &viewer = NBodyGLViewer::getInstance();



		YAML::Node &config = root["NBodyConfig"];
		if(!viewer.init(&argc, argv, resolver, config)){
			cout<<"Initialization failure"<<endl;
			system("PAUSE");
			return 1;
		}

		viewer.start();


		FreeLibrary(module);
	}catch(YAML::Exception &e){
		cout<<e.what()<<endl;
		system("PAUSE");
		return 1;
	}

	cout<<"Finished execution"<<endl;
	system("PAUSE");
	return 0;
}
