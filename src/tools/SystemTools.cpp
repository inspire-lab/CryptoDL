/*
 * SystemTools.cpp
 *
 *  Created on: Jun 5, 2019
 *      Author: robert
 */


#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include "SystemTools.h"


int executePython( std::string pythonCode ){
	const std::string python = "python -c ";
	const std::string python3 = "python3 -c ";
	std::string command;
	if( USE_PYTHON_3 )
		command = python3 + pythonCode;
	else
		command = python + pythonCode;


	std::cout << "Python call: " << command << std::endl;

	// run python code
	int r = system( command.c_str() );
	if ( r != EXIT_SUCCESS  ){
		std::cerr <<  "python processing messed up" << std::endl;
	}
	return r;
}


std::string getCurrentWorkingDir() {
  char buff[FILENAME_MAX];
  getcwd( buff, FILENAME_MAX );
  std::string current_working_dir(buff);
  return current_working_dir;
}
