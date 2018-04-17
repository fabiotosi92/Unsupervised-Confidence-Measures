#include "census.hpp"

using namespace cv;
using namespace std;

uchar compare_value_binary
(
	uchar center, 
	uchar other
){
	uchar census = 0, bit = 0;

	if (center >= other)
	{
		bit = 1;
		census <<= 1;
		census += bit;
	}
	else
	{
		bit = 0;
		census <<= 1;
		census += bit;
	}
	
	return census;
}

void census_transform_binary
(
	InputArray src, 
	int r, 
	OutputArray dst
){
	Mat _src = src.getMat();
	int window_size = 2 * r + 1;
	int height = _src.rows;
	int width = _src.cols;

	switch (r)
	{
		case 1:
			dst.create(_src.size(), CV_8U);
			break;
		case 2:
			dst.create(_src.size(), CV_32F);
			break;
		default:
			cout << "Unsupported window size for census trasform!" << endl;
			break;
	}

	Mat _dst = dst.getMat();

	long census, bit, centerCount;
	uchar center, left, right, up, down;

	for (int row = r; row < height - r; row++)
	{
		for (int col = r; col < width - r; col++)
		{
			census = 0;
			centerCount = 0;

			for (int x = row - r; x <= row + r; x++)
			{
				for (int y = col - r; y <= col + r; y++)
				{
					if (centerCount != window_size * window_size / 2) 
					{
						census <<= 1; 
						bit = compare_value_binary(_src.ptr<uchar>(row)[col], _src.ptr<uchar>(x)[y]);
						census += bit; 
					}

					centerCount++; 
				}
			}
			
			if (_dst.depth() == CV_8U)
				_dst.ptr<uchar>(row)[col] = census;
			else
				_dst.ptr<int>(row)[col] = census;
		}
	}
}

int hamming_distance_bis
(
	long left, 
	long right
){

	int distance = 0;
	int val = left ^ right;

	while (val != 0)
	{
		val = val & (val - 1);
		distance++;
	}

	return distance;
}