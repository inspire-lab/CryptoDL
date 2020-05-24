/*
 * RNNTools.h
 *
 *  Created on: May 29, 2019
 *      Author: robert
 */

#ifndef TOOLS_RNNTOOLS_H_
#define TOOLS_RNNTOOLS_H_


#include <string>
#include <vector>


/**
 * @brief
 *
 *
 */
std::pair<std::vector<float>,std::vector<int8_t>> getEmbeddings( const std::string& modelFile, int maxlen, int maxwords, std::string dataset, std::string outDir = "" );



#endif /* TOOLS_RNNTOOLS_H_ */
