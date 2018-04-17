#ifndef SORT_H
#define SORT_H

#include <opencv2/opencv.hpp>

void MergeSort
(
	float confidence[], 
	float rows_mem[], 
	float cols_mem[], 
	int p, int r, 
	int size
);

void reverse
(
	float* confidences, 
	float rows_mem[], 
	float cols_mem[], 
	int len
);

#endif
