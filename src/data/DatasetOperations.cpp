#include <boost/algorithm/string/predicate.hpp>
#include "DatasetOperations.h"
#include "../tools/FileSystemTools.h"
#include "../tools/Config.h"


/*** DATASET OPERATION ***/

mnist::MNIST_dataset<uint8_t, uint8_t> loadMNIST(const std::string& path){
    // Load MNIST data
    mnist::MNIST_dataset<uint8_t, uint8_t> dataset = mnist::read_dataset(path);
    return dataset;
}


mnist::MNIST_dataset<uint8_t, uint8_t> loadMNIST(){
    // Load MNIST data
    return loadMNIST( Config::getConfig()->get<std::string>( "datasets", "mnist-home" ) );
}


std::pair<std::vector<float_flat_img>, std::vector<u_int8_t>> loadCOWC( const std::string& path, float quantFactor ){
	std::vector<float_flat_img> imgs;
	std::vector<u_int8_t> labels;

	auto content = readDirectory( path );
	for( auto entry: content ){
		auto dirEntry = joinPath( path, entry );
		if( isDir( dirEntry ) ){
			auto testDir = joinPath( dirEntry, "test" );
			auto testContent = readDirectory( testDir );
			int count = 0;
			std::cout << "reading " << testDir;
			for( auto imgName : testContent ){
				++count;
				auto imgFile = joinPath( testDir, imgName );
				if( !isFile( imgFile ) || ! boost::ends_with( imgFile, ".jpg" )  )
					continue;
				std::cout << "\r " << count << "/" << testContent.size();
				imgs.push_back( readJPGquantizedFlat( imgFile, quantFactor ) );
				labels.push_back( 0 ? boost::starts_with( imgName, "neg" ) : 1 );
			}
			std::cout << std::endl;
		}
	}

	return std::pair<std::vector<float_flat_img>, std::vector<u_int8_t>>( imgs, labels );
}


