/*
 * IOStream.h
 *
 *  Created on: May 23, 2020
 *      Author: robert
 */

#ifndef TOOLS_IOSTREAM_H_
#define TOOLS_IOSTREAM_H_

#include <iostream>
#include <vector>


// FIXME needed with a newer version of HELIB
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T> &input) {
	os << "[";
	for( auto it = input.begin(); it != input.end(); ++it ){
		os << " " << *it;
		if( std::next( it ) != input.end() )
			os << ",";
	}
	os << "]";
	return os;
}



#endif /* TOOLS_IOSTREAM_H_ */
