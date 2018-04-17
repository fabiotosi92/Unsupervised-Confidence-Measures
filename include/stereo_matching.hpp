#ifndef STEREO_H
#define STEREO_H

#include "DSI.hpp"
#include "box_filter.hpp"

_DSI SHD_box_filtering
(
	cv::InputArray left, 
	cv::InputArray right, 
	int d_min, 
	int d_max, 
	int r_CENSUS, 
	int r_BOX_FILTER
);

#endif

