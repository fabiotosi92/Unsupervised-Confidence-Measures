#ifndef DSI_H
#define DSI_H

#include <opencv2/opencv.hpp>

typedef struct _DSI
{
	int width;
	int height;
	int d_max;
	int d_min;
	int num_disp;
	bool similarity;
	std::vector <cv::Mat> values;

}_DSI;

_DSI DSI_init
(
	int height, 
	int width, 
	int d_min, 
	int d_max, 
	bool similarity
);

_DSI DSI_left2right
(
	_DSI DSI_L
);

cv::Mat disparity_map_L2R
(
	_DSI DSI
);

cv::Mat disparity_map_R2L
(
	_DSI DSI
);

void write_disparity_map
(
	cv::InputArray disparity_map, 
    cv::InputArray mask, 
	std::string file_name
);

#endif
