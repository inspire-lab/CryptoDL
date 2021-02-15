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
#include <stdexcept>
#include "DataReaders.h"


/**
 * @brief
 *
 *
 */
std::pair<std::vector<float>,std::vector<int8_t>> getEmbeddings( const std::string& modelFile, int maxlen, int maxwords, std::string dataset, std::string outDir = "" );



class Embedding{
    public:
    std::vector<std::vector<float>> embeddingMatrix;
    const uint embeddingDim;
    const uint inputDim;

    Embedding( uint embeddingDim, uint inputDim, const std::string& file  );

    std::vector<float> embed( const std::vector<std::vector<int>>& idx, int batchSize = -1 );

    /**
     * Transform word indices into embeddings.
     */
    std::vector<float> embed( const std::vector<int>& idx, int batchSize = -1 );
};


#endif /* TOOLS_RNNTOOLS_H_ */
