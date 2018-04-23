#include "DSI.hpp"
#include "census.hpp"
#include "box_filter.hpp"

using namespace cv;

_DSI hamming_distance_cost
(
	InputArray left, 
	InputArray right, 
	int d_min, 
	int d_max, 
	int r
){
	Mat _left = left.getMat();
	Mat _right = right.getMat();
	Mat _left_census, _right_census;

	int height = _left.rows;
	int width = _left.cols;

	int num_disp = d_max - d_min + 1;

	/*DSI Initialization*/
	_DSI DSI = DSI_init(_left.rows, _right.cols, d_min, d_max, 0);

	census_transform_binary(left, r,  _left_census);
	census_transform_binary(right, r, _right_census);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			for (int d = d_min; d < d_max + 1; d++)
			{
				int cost;
				int col_R = (col - d < 0 || col - d >= _left.cols) ? 0 : col - d;
				
				if (r == 1)
					cost = hamming_distance_bis(_left_census.ptr<uchar>(row)[col], _right_census.ptr<uchar>(row)[col_R]);
				else
					cost = hamming_distance_bis(_left_census.ptr<int>(row)[col], _right_census.ptr<int>(row)[col_R]);
				
				DSI.values[d - d_min].ptr<float>(row)[col] = cost;
			}
		}
	}

	return DSI;
}

_DSI SHD_box_filtering
(
	InputArray left, 
	InputArray right, 
	int d_min, 
	int d_max, 
	int r_CENSUS, 
	int r_BOX_FILTER
){
	int num_disp = d_max - d_min + 1;

	Mat _left, _right;

	_left = left.getMat();
	_right = right.getMat();

	//DSI initialization
	_DSI DSI = DSI_init(_left.rows, _left.cols, d_min, d_max, 0);
	_DSI DSI_CENSUS = hamming_distance_cost(_left, _right, d_min, d_max, r_CENSUS);

	for (int d = 0; d < num_disp; d++)
	{
		box_filter(DSI_CENSUS.values[d], r_BOX_FILTER, DSI.values[d]);
		DSI.values[d] /= ((5 * 5 - 1)*(5 * 5));
	}
		
	return DSI;
}


