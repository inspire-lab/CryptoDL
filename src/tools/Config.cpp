/*
 * Config.cpp
 *
 *  Created on: Jul 2, 2019
 *      Author: robert
 */


#include "Config.h"
#include "SystemTools.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cctype>


Config* Config::mConfig = NULL;

Config* Config::getConfig(){
	if( !mConfig ){
		mConfig = new Config;
		mConfig->readConfig();
	}
	return mConfig;
}

void Config::readConfig(){
	std::cout << "loading config file " << getCurrentWorkingDir() << "/" << Config::configFile << std::endl;
	std::ifstream file( Config::configFile );

	if( !file ){
		std::cerr << "can't find config file";
		exit(1);
	}

	std::string str;
	config c;
	config_section* s;
	while ( std::getline( file, str ) ){
		// remove whitespaces
		str.erase( remove_if( str.begin(), str.end(), isspace ), str.end() );

		// is it a comment
		if( boost::starts_with( str, "#" ) )
			continue;

		// is it an empty line, or is the line too short?
		// we need at least 3 characters
		if( str.size() < 3 )
			continue;

		// start section
		if( boost::starts_with( str, "[" ) ){
			if( ! boost::ends_with( str, "]" ) )
				throw new std::runtime_error( str + " is not a valid section header" );
			//remove []
			str = str.substr( 1, str.size() - 2 );
			c[ str ] = config_section();
			s = &c[ str ];
			std::cout << str << std::endl;
		}
		// read section content;
		else {
			// needs to have a "="
			if( str.find( '=' ) == std::string::npos )
				throw new std::runtime_error( str + " is not a valid config entry" );

			// split into key value
			std::vector<std::string> key_value;
			boost::split( key_value, str, boost::is_any_of( "=" ) );

			// more than one "=" is illegal
			if( key_value.size() != 2 )
				throw new std::runtime_error( str + " is not a valid config entry" );

			// assign into our current section
			s->operator []( key_value[ 0 ] ) = key_value[ 1 ];

			std::cout << "\t" << key_value[ 0 ] << "=" << key_value[ 1 ] << std::endl;
		}

	}
	file.close();
	mConfig->c = c;
}


bool toBool( const std::string& in  ){
	std::string str( in );
    std::transform( str.begin(), str.end(), str.begin(), ::tolower );
    std::istringstream is( str );
    bool b;
    is >> std::boolalpha >> b;
    return b;
}


template<>
bool Config::get<bool>( const std::string& section, const std::string& key ){
	// check if the section exists
	if ( Config::c.count( section ) < 1 )
		return false;

	// check if the key exists
	if ( Config::c[ section ].count( key ) < 1 )
			return false;

	return toBool( Config::c[ section ][ key ] );
}

template<>
std::string Config::get<std::string>( const std::string& section, const std::string& key ){
	// check if the section exists
	if ( Config::c.count( section ) < 1 )
		return "";

	// check if the key exists
	if ( Config::c[ section ].count( key ) < 1 )
			return "";

	return Config::c[ section ][ key ];
}

template<>
long Config::get<long>( const std::string& section, const std::string& key ){
	// check if the section exists
	if ( Config::c.count( section ) < 1 )
		return 0;

	// check if the key exists
	if ( Config::c[ section ].count( key ) < 1 )
			return 0;

	return std::stol( Config::c[ section ][ key ] );
}


template<>
double Config::get<double>( const std::string& section, const std::string& key ){
	// check if the section exists
	if ( Config::c.count( section ) < 1 )
		return 0;

	// check if the key exists
	if ( Config::c[ section ].count( key ) < 1 )
			return 0;

	return std::stod( Config::c[ section ][ key ] );
}





