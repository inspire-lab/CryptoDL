/*
 * FileSystemTools.cpp
 *
 *  Created on: May 23, 2019
 *      Author: robert
 */
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <cstdarg>
#include <boost/filesystem.hpp>
#include "FileSystemTools.h"


bool fileExists(const std::string& file) {
    struct stat buf;
    return ( stat( file.c_str(), &buf ) == 0 );
}


std::string getFileName( const std::string& filePath, bool withExtension, char seperator ){
	// Get last dot position
	std::size_t dotPos = filePath.rfind('.');
	std::size_t sepPos = filePath.rfind(seperator);
	if( sepPos != std::string::npos )
		return filePath.substr( sepPos + 1, filePath.size() - ( withExtension || dotPos != std::string::npos ? 1 : dotPos ) );
	return "";
}


std::string getDirectory( const std::string& filePath, char seperator ){
	if( isDir( filePath.c_str() ) )
		return filePath;
	std::size_t  sepPos = filePath.rfind( seperator );
	return filePath.substr( 0, std::min( sepPos , filePath.size() ) );
}

std::vector<std::string> readDirectory(const std::string& path )
{
	std::vector<std::string> ret;
    boost::filesystem::path p( path );
    boost::filesystem::directory_iterator end;
    for( boost::filesystem::directory_iterator iter( p ); iter != end; ++iter ){
    	ret.push_back( iter->path().leaf().string() );
    }
    return ret;
}

bool isFile(const char* path) {
    struct stat buf;
    stat(path, &buf);
    return S_ISREG( buf.st_mode );
}

bool isDir(const char* path) {
    struct stat buf;
    stat(path, &buf);
    return S_ISDIR( buf.st_mode );
}

std::string joinPath( const std::string& path , const std::string& entry){
	if( path.size() == 0 )
		return entry;
	return path + "/" + entry;
}


