#include "confidence_measures_utility.hpp"


/***************************************************************************************************/
/*
* Compute the minimum of each column of the cost volume 
* c1_idx is the index of the minimum in the column
*
/***************************************************************************************************/
void minimum(InputArrayOfArrays costs, 
	         OutputArray c1, 
	         OutputArray c1_idx)
{
	vector<Mat> _costs;
	costs.getMatVector(_costs);

	int height = _costs[0].rows;
	int width = _costs[0].cols;
	int num_disp = _costs.size();

	c1.create(Size(width, height), CV_32F);
	c1_idx.create(Size(width, height), CV_8U);

	Mat min = c1.getMat();
	Mat idx = c1_idx.getMat();

	min.setTo(Scalar(-1));
	idx.setTo(Scalar(-1));

	float minimum;
	int index;

	for (int row = 0; row < height; row++)
	{
		float* min_ptr = min.ptr<float>(row);
		uchar* idx_ptr = idx.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			minimum = FLT_MAX;
			index = -1;
			
			for (int d = 0; d < num_disp; d++)
			{
				float value = _costs[d].ptr<float>(row)[col];

				if (value < minimum && value >= 0)
				{
					minimum = value;
					index = d;
				}
			}
			
			if (index >= 0)
			{
				min_ptr[col] = minimum;
				idx_ptr[col] = index;
			}
		}
	}
}

void minimum(InputArrayOfArrays costs, 
	         OutputArray c1)
{
	vector<Mat> _costs;
	costs.getMatVector(_costs);

	int height = _costs[0].rows;
	int width = _costs[0].cols;
	int num_disp = _costs.size();

	c1.create(Size(width, height), CV_32F);

	Mat min = c1.getMat();

	min.setTo(Scalar(-1));

	float minimum;
	int index;

	for (int row = 0; row < height; row++)
	{
		float* min_ptr = min.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			minimum = FLT_MAX;
			index = -1;

			for (int d = 0; d < num_disp; d++)
			{
				float value = _costs[d].ptr<float>(row)[col];

				if (value < minimum && value >= 0)
				{
					minimum = value;
					index = d;
				}
			}
		
			if (index >= 0)
			{
				min_ptr[col] = minimum;
			}
		}
	}
}

/***************************************************************************************************/
/*
* Compute the sum of each column of the cost volume 
*
/***************************************************************************************************/
void sum(InputArrayOfArrays costs, 
	     OutputArray sum)
{
	vector<Mat> _costs;
	costs.getMatVector(_costs);

	int height = _costs[0].rows;
	int width = _costs[0].cols;
	int num_disp = _costs.size();

	sum.create(Size(width, height), CV_32F);
	Mat _sum = sum.getMat();
	_sum.setTo(Scalar(-1));

	float sum_cost;

	for (int row = 0; row < height; row++)
	{
		float* _sum_ptr = _sum.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			sum_cost = 0;

			for (int d = 0; d < num_disp; d++)
			{
				float cost = _costs[d].ptr<float>(row)[col];

				sum_cost += cost;
			}

			_sum_ptr[col] = sum_cost;
		}
	}
}

/***************************************************************************************************/
/*
* Compute the local minima of the cost volume filter for each column
* No smoothing is applied
* The output volume contains 1 if local minima 0 otherwise (CV_8U)
*
/***************************************************************************************************/
void inflections(InputArrayOfArrays costs, 
	             OutputArrayOfArrays local_minima)
{
	vector <Mat> _local_minima,
	             _costs;
	costs.getMatVector(_costs);

	int height = _costs[0].rows;
	int width = _costs[0].cols;
	int num_disp = _costs.size();

	for (int i = 0; i < num_disp; i++)
	{
		_local_minima.push_back(Mat(height, width, CV_8U, Scalar(0)));
	}

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			for (int d = 0; d < num_disp; d++)
			{
				float central = _costs[d].ptr<float>(row)[col];

				if (central >= 0)
				{
					if (d == 0)
					{
						float right = _costs[d + 1].ptr<float>(row)[col];

						if (central < right && right >= 0)
							_local_minima[d].ptr<uchar>(row)[col] = 1;
					}
					else if (d == num_disp - 1)
					{
						float left = _costs[d - 1].ptr<float>(row)[col];

						if (central < left && left>=0)
							_local_minima[d].ptr<uchar>(row)[col] = 1;
					}
					else
					{
						float right = _costs[d + 1].ptr<float>(row)[col];
						float left = _costs[d - 1].ptr<float>(row)[col];

						if ((central < right) && (central < left) && (left >= 0) && right >= 0)
							_local_minima[d].ptr<uchar>(row)[col] = 1;
					}
				}
			}
		}
	}

	//local minima
	local_minima.create(_local_minima.size(), 1, Mat(_local_minima).type());

	for (int i = 0; i < _local_minima.size(); i++)
	{
		local_minima.getMatRef(i) = _local_minima[i];
	}
	
}

/***************************************************************************************************/
/*
* Compute the base costs such as c1, c2, c_hat_2, inflections
*
/***************************************************************************************************/
void compute_base_costs(InputArrayOfArrays costs_DSI, 
						OutputArray c1, 
						OutputArray c1_idx,
						OutputArray c2, 
						OutputArray c2_idx, 
						OutputArray c_hat_2, 
						OutputArray c_hat_2_idx, 
						OutputArray c_sum, 
						OutputArray NOI,
						OutputArrayOfArrays local_minima)
{
	vector<Mat> values, _inflections; costs_DSI.getMatVector(values);
	vector<Mat> costs = copy(values);

	int height = costs[0].rows;
	int width = costs[0].cols;
	int num_disp = costs.size();

	//compute the first minimum.
	cout << " - compute the first minimum..." << endl;
	minimum(costs, c1, c1_idx); 

	//compute the sum of matching costs.
	cout << " - compute the sum of matching costs..." << endl;
	sum(costs, c_sum);

	//index to all the local minima
	cout << " - index to all the local minima..." << endl; 

	inflections(costs, local_minima);
	local_minima.getMatVector(_inflections);

	//compute the second minimum.
	cout << " - compute the second minimum..." << endl;
	for (int row = 0; row < height; row++)
	{
		uchar* c1_idx_ptr = c1_idx.getMat().ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			uchar index = c1_idx_ptr[col];
			costs[index].ptr<float>(row)[col] = FLT_MAX;
		}
	}

	minimum(costs, c2, c2_idx);

	//compute the second 'local' minimum.
	cout << " - compute the second 'local' minimum..." << endl;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			for (int d = 0; d < num_disp; d++)
			{
				if (_inflections[d].ptr<uchar>(row)[col] == 0)
					costs[d].ptr<float>(row)[col] = FLT_MAX;
			}
		}
	}

	minimum(costs, c_hat_2, c_hat_2_idx);

	NOI.create(Size(costs[0].cols, costs[0].rows), CV_32F);
	Mat _NOI = NOI.getMat();

	//compute the number of inflection (NOI) points.
	cout << " - compute the number of inflection(NOI) points..." << endl;
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			float number_of_inflections = 0;
			float* NOI_ptr = _NOI.ptr<float>(row);

			for (int d = 0; d < num_disp; d++)
			{
				if (_inflections[d].ptr<uchar>(row)[col] == 1)
					number_of_inflections++;
			}

			NOI_ptr[col] = - number_of_inflections;
		}
	}
}

/***************************************************************************************************/
/*
* Useful function to normalize a vector of Mat
*
/***************************************************************************************************/
vector<Mat> normalize(InputArrayOfArrays input)
{
	vector<Mat> _normalized_vector, _input;
	input.getMatVector(_input);

	for (int d = 0; d < _input.size(); d++)
	{
		Mat _normalized;
		normalize(_input[d], _normalized, 0, 1, NORM_MINMAX, CV_32F);
		_normalized_vector.push_back(_normalized);
	}

	return _normalized_vector;
}

/***************************************************************************************************/
/*
* Useful function to copy a vector of Mat
*
/***************************************************************************************************/
vector<Mat> copy(vector<Mat> values)
{
	vector<Mat> _copy;

	for (int d = 0; d < values.size(); d++)
	{
		Mat c; values[d].copyTo(c);
		_copy.push_back(c);
	}

	return _copy;
}

/***************************************************************************************************/
/*
* Useful function to find the confidence to process from a vector of string
*
/***************************************************************************************************/
bool use(vector<string> confidences, 
	     string confidence)
{
	return find(confidences.begin(), confidences.end(), confidence) != confidences.end();
}