#ifndef _CONFIDENCE_MEASURES_UTILITY
#define _CONFIDENCE_MEASURES_UTILITY

#include "DSI.hpp"
#include "stereo_matching.hpp"
#include <fstream>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/***************************************************************************************************/
/*
* Compute the minimum of each column of the cost volume
* c1_idx is the index of the minimum in the column
*
/***************************************************************************************************/
void minimum
(
	InputArrayOfArrays costs, 
	OutputArray c1, 
	OutputArray c1_idx
);

void minimum
(
	InputArrayOfArrays costs, 
	OutputArray c1
);

/***************************************************************************************************/
/*
* Compute the sum of each column of the cost volume
*
/***************************************************************************************************/
void sum
(
	InputArrayOfArrays costs, 
	OutputArray sum
);

/***************************************************************************************************/
/*
* Compute the local minima of the cost volume filter for each column
* No smoothing is applied
* The output volume contains 1 if local minima 0 otherwise (CV_8U)
*
/***************************************************************************************************/
void inflections
(
	InputArrayOfArrays costs, 
	OutputArrayOfArrays local_minima
);

/***************************************************************************************************/
/*
* Compute the base costs such as c1, c2, c_hat_2, inflections
*
/***************************************************************************************************/
void compute_base_costs
(
	InputArrayOfArrays costs_DSI, 
	OutputArray c1, 
	OutputArray c1_idx, 
	OutputArray c2, 
	OutputArray c2_idx, 
	OutputArray c_hat_2, 
	OutputArray c_hat_2_idx, 
	OutputArray c_sum, 
	OutputArray NOI, 
	OutputArrayOfArrays local_minima
);

/***************************************************************************************************/
/*
* Utility functions to save a confidence map or a vector of confidence maps
* Normalization is applied for each confidence map
* Histogram equalization is used for better visualization
*
/***************************************************************************************************/
void save_confidence
(
	InputArray confidence_map, 
	string filename
);

void save_confidences
(
	InputArrayOfArrays confidence, 
	vector<string> methods, 
	string path
);

/***************************************************************************************************/
/*
* Useful function to normalize a vector of Mat
*
/***************************************************************************************************/
vector<Mat> normalize
(
	InputArrayOfArrays input
);

/***************************************************************************************************/
/*
* Useful function to copy a vector of Mat
*
/***************************************************************************************************/
vector<Mat> copy
(
	vector<Mat> values
);

/***************************************************************************************************/
/*
* Useful function to find the confidence to process from a vector of string
*
/***************************************************************************************************/
bool use
(
	vector<string> confidences, 
	string confidence
);

#endif 
