#include "DSI.hpp"

using namespace cv;
using namespace std;

_DSI DSI_init(int height, int width, int d_min, int d_max,  bool similarity)
{
	struct _DSI DSI;

	DSI.height = height;
	DSI.width = width;
	DSI.d_min = d_min;
	DSI.d_max = d_max;
	DSI.num_disp = d_max - d_min + 1;
	DSI.similarity = similarity;

	for (int d = 0; d < d_max - d_min + 1; d++)
	{
		DSI.values.push_back(Mat(height, width, CV_32F, Scalar(-1)));
	}

	return DSI;
}

_DSI DSI_left2right(_DSI DSI_L)
{
	int height = DSI_L.height;
	int width = DSI_L.width;
	int d_min = DSI_L.d_min;
	int d_max = DSI_L.d_max;
	int similarity = DSI_L.similarity;
	int num_disp = d_max - d_min + 1;

	_DSI DSI_R = DSI_init(height, width, d_min, d_max, similarity);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			for (int d = 0; d < num_disp; d++)
			{
				if (col + d < width) 
					DSI_R.values[d].ptr<float>(row)[col] = DSI_L.values[d].ptr<float>(row)[col + d];
			}
		}
	}

	return DSI_R;
}

Mat disparity_map_L2R(_DSI DSI)
{
	Mat disparity = Mat(DSI.height, DSI.width, CV_32F);

	float min_max;
	int WTA;

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			min_max = (DSI.similarity) ? 0 : FLT_MAX;
			WTA = 0;

			for (int d = 0; d < DSI.num_disp; d++)
			{
				float matching_cost = DSI.values[d].ptr<float>(row)[col]; 
				
				if (!DSI.similarity && min_max > matching_cost)
				{
					min_max = matching_cost; 
					WTA = d + DSI.d_min; 
				}
				else if (DSI.similarity && min_max < matching_cost)
				{
					min_max = matching_cost;
					WTA = d + DSI.d_min;
				}
			}

			if (WTA > 0) disparity.ptr<float>(row)[col] = (float)WTA; 
			else  disparity.ptr<float>(row)[col] = 0;
		}
	}

	return disparity;
}

Mat disparity_map_R2L(_DSI DSI)
{
	Mat disparity = Mat(DSI.height, DSI.width, CV_32F);

	float min_max;
	int WTA;

	for (int row = 0; row < DSI.height; row++)
	{
		for (int col = 0; col < DSI.width; col++)
		{
			min_max = (DSI.similarity) ? 0 : FLT_MAX;
			WTA = 0;

			for (int d = 0; d < DSI.num_disp && col + d < DSI.width; d++)
			{
				float matching_cost = DSI.values[d].ptr<float>(row)[col + d];

				if (!DSI.similarity && min_max > matching_cost)
				{
					min_max = matching_cost;
					WTA = d + DSI.d_min;
				}
				else if (DSI.similarity && min_max < matching_cost)
				{
					min_max = matching_cost;
					WTA = d + DSI.d_min;
				}
			}

			if (WTA > 0) disparity.at<float>(row, col) = WTA;
			else  disparity.at<float>(row, col) = 0;
		}
	}

	return disparity;
}

void write_disparity_map(InputArray disparity_map, InputArray mask, string file_name)
{
	Mat _mask, _temp = disparity_map.getMat();
	Mat _disparity_map; _disparity_map.create(_temp.size(), CV_8U);

	if (mask.empty())
	{
		_mask = Mat(_disparity_map.size(), CV_32F);
		_mask.setTo(255);
	}
	else
		_mask = mask.getMat();

	int height = _disparity_map.rows;
	int width = _disparity_map.cols;

	for (int row = 0; row < height; row++)
	{
		uchar* disparity_map_ptr = _disparity_map.ptr<uchar>(row);
		float* temp_ptr = _temp.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			if (temp_ptr[col] >= 0 && _mask.ptr<float>(row)[col] == 255)
				disparity_map_ptr[col] = (uchar)temp_ptr[col];
			else
				disparity_map_ptr[col] = 0;
		}
	}

	imwrite(file_name, _disparity_map);
}

