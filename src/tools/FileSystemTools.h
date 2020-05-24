/*
 * FileSystemTools.h
 *
 *  Created on: May 23, 2019
 *      Author: robert
 */

#ifndef TOOLS_FILESYSTEMTOOLS_H_
#define TOOLS_FILESYSTEMTOOLS_H_

#include <vector>

/**
 * Check if a file exists
 * @return true if and only if the file exists, false else
 */
bool fileExists(const std::string& file);


/**
 * @brief Get File Name from a Path with or without extension
 *
 *
 * taken from
 * https://thispointer.com/c-how-to-get-filename-from-a-path-with-or-without-extension-boost-c17-filesytem-library/
 */
std::string getFileName(const std::string& filePath, bool withExtension = true, char seperator = '/');

/**
 * @brief Remove filename from a path. If the last element in the path is a directory
 * returns the unaltered path.
 *
 */
std::string getDirectory(const std::string& filePath, char seperator = '/');

/**
 * @brief Returns the contents of a direcorty
 */
std::vector<std::string> readDirectory(const std::string& path );


/**
 * @brief Checks if given path points to a file
 */
bool isFile(const char* path);
inline bool isFile(const std::string& path){
	return isFile( path.c_str() );
}


/**
 * @brief Checks if given path points to a directory
 */
bool isDir(const char* path);
inline bool isDir(const std::string& path){
	return isDir( path.c_str() );
}

/**
 * @brief returns a '/' seperated path
 */

std::string joinPath( const std::string& path , const std::string& entry);



#endif /* TOOLS_FILESYSTEMTOOLS_H_ */
