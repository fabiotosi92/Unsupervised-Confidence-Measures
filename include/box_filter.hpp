#ifndef BOXFILTER_H
#define BOXFILTER_H

#include <opencv2/opencv.hpp>

void cumsum
(	
	cv::InputArray imSrc, 
	int n, 
	cv::OutputArray imDst
);

void box_filter
(	
	cv::InputArray imSrc, 
	int r, 
	cv::OutputArray imDst
);

#endif