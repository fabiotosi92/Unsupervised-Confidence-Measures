#ifndef CENSUS_H
#define CENSUS_H

#include <opencv2/opencv.hpp>

void census_transform_binary
(
	cv::InputArray src, 
	int r, 
	cv::OutputArray dst
);

int hamming_distance_bis
(
	long left, 
	long right
);

#endif 
