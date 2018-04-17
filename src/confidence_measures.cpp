#include "confidence_measures.hpp"

void set_parameters
(
	confParams &params
)
{
	params.lc_gamma = 480;
	params.pkr_epsilon = 0.1;
	params.lrd_epsilon = 0.1;
	params.apkr_radius = 12;
	params.apkr_epsilon = 0.01;
	params.per_sigma = 120;
	params.mlm_sigma = 2;
	params.aml_sigma = 2;
	params.nmm_sigma = 2;
	params.zsad_radius = 2;
	params.med_radius = 5;
	params.dsm_epsilon = 0.001;
	params.DD_edge_threshold = 1;
	params.DD_radius = 1;
	params.DD_max_lowThreshold = 50;
	params.DD_ratio = 3;
	params.lmn_radius = 2;
	params.dvm = 3;
}

/***************************************************************************************************/
/*
* Confidence Measures
*
/***************************************************************************************************/

//2.1   Matching Cost
void matching_score_measure
(
	InputArray c1, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1 = c1.getMat(),
		_confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		float* c1_ptr = _c1.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			float c_1 = c1_ptr[col];

		    confidence_map_ptr[col] = - c_1;
		}
	}
}

//2.2	Local properties of the cost curve
void curvature
(
	InputArrayOfArrays costs, 
	InputArray c1_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
		_confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	int d1;
	float minimum, add1, add2;

	for (int row = 0; row < height; row++)
	{
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);
		
		for (int col = 0; col < width; col++)
		{
			d1 = c1_idx_ptr[col]; 
			minimum = _costs[d1].ptr<float>(row)[col];

			if (d1 - 1 < 0)
			{
				add2 = _costs[d1 + 1].ptr<float>(row)[col];
				add1 = _costs[d1 + 1].ptr<float>(row)[col];
			}
			else if (d1 + 1 > _costs.size() - 1)
			{
				add1 = _costs[d1 - 1].ptr<float>(row)[col];
				add2 = _costs[d1 - 1].ptr<float>(row)[col];
			}
			else
			{
				add1 = _costs[d1 - 1].ptr<float>(row)[col];
				add2 = _costs[d1 + 1].ptr<float>(row)[col];
			}

			confidence_map_ptr[col] = (-2 * minimum) + add1 + add2;		
		}
	}
}

void local_curve
(
	InputArrayOfArrays costs, 
	InputArray c1_idx, 
	float gamma, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
		_confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	int d1;
	float minimum, add1, add2, max;

	for (int row = 0; row < height; row++)
	{
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			d1 = c1_idx_ptr[col];
			minimum = _costs[d1].ptr<float>(row)[col];

			if (d1 - 1 < 0)
			{
				add2 = _costs[d1 + 1].ptr<float>(row)[col];
				add1 = _costs[d1 + 1].ptr<float>(row)[col];
			}
			else if (d1 + 1 > _costs.size() - 1)
			{
				add1 = _costs[d1 - 1].ptr<float>(row)[col];
				add2 = _costs[d1 - 1].ptr<float>(row)[col];
			}
			else
			{
				add1 = _costs[d1 - 1].ptr<float>(row)[col];
				add2 = _costs[d1 + 1].ptr<float>(row)[col];
			}

			if (add1 >= add2) max = add1;
			else max = add2;

			confidence_map_ptr[col] = (max - minimum) / gamma;
		}
	}
}

//2.3	Local minima of the cost curve
void peak_ratio
(
	InputArray c1, 
	InputArray c2m, 
	float epsilon, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1 = c1.getMat(), 
	    _c2m = c2m.getMat(),
		_confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		float* c1_ptr = _c1.ptr<float>(row);
		float* c2m_ptr = _c2m.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			float c_1 = c1_ptr[col];
			float c_2m = c2m_ptr[col];

			confidence_map_ptr[col] = (c_2m + epsilon) / (c_1 + epsilon); 	
		}
	}
}

void average_peak_ratio
(
	InputArrayOfArrays costs, 
	InputArray c1_idx, 
	InputArray c2m_idx, 
	float epsilon, 
	int r, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
	    _c2m_idx = c2m_idx.getMat(),
	    _confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	for (int row = 0; row < height; row++)
	{
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);
		uchar* c2m_idx_ptr = _c2m_idx.ptr<uchar>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			uchar c_1_idx = c1_idx_ptr[col];
			uchar c_2m_idx = c2m_idx_ptr[col];
			float sum = 0;
			
			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					//c_2m_idx >= 0 if second local minimum doesn't exist
					if (row + i >= 0 && row + i < height && col + j >= 0 && col + j < width && c_2m_idx >= 0) 
					{
						//q is the running pixel 
						float q_c1 = _costs[c_1_idx].ptr<float>(row + i)[col + j];
						float q_c2m = _costs[c_2m_idx].ptr<float>(row + i)[col + j];

						sum += (q_c2m + epsilon) / (q_c1 + epsilon);
					}
				}
			}

			confidence_map_ptr[col] = sum / (2 * r + 1) * (2 * r + 1); //average
		}
	}
}

void maximum_margin
(
	InputArray c1, 
	InputArray c2m, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1 = c1.getMat(),
	    _c2m = c2m.getMat(), 
	    _confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		float* c1_ptr = _c1.ptr<float>(row);
		float* c2m_ptr = _c2m.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			float c_1 = c1_ptr[col];
			float c_2m = c2m_ptr[col];

			confidence_map_ptr[col] = c_2m - c_1;
		}
	}
}

void non_linear_margin
(
	InputArray c1, 
	InputArray c2m, 
	float sigma, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1 = c1.getMat(),
		_c2m = c2m.getMat(),
		_confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		float* c1_ptr = _c1.ptr<float>(row);
		float* c2m_ptr = _c2m.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			float c_1 = c1_ptr[col];
			float c_2m = c2m_ptr[col];

			confidence_map_ptr[col] = exp(c_2m - c_1) / (2 * (sigma * sigma));
		}
	}
}

void disparity_variance_measure
(
	InputArray disparity_map, 
	int size, 
	OutputArray confidence_map
){
	confidence_map.create(disparity_map.size(), CV_32F);

	Mat _disparity_map = disparity_map.getMat(),
		_confidence_map = confidence_map.getMat(),
		_confidence_measure_x = Mat(disparity_map.size(), CV_32F),
		_confidence_measure_y = Mat(disparity_map.size(), CV_32F),
		_abs_confidence_measure_x,
		_abs_confidence_measure_y;

	Sobel(disparity_map, _confidence_measure_x, disparity_map.depth(), 1, 0, size, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(_confidence_measure_x, _abs_confidence_measure_x);

	Sobel(disparity_map, _confidence_measure_y, disparity_map.depth(), 0, 1, size, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(_confidence_measure_y, _abs_confidence_measure_y);

	for (int row = 0; row < _disparity_map.rows; row++)
	{
		uchar* abs_confidence_measure_x_ptr = _abs_confidence_measure_x.ptr<uchar>(row);
		uchar* abs_confidence_measure_y_ptr = _abs_confidence_measure_y.ptr<uchar>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < _disparity_map.cols; col++)
		{
			uchar x_value = abs_confidence_measure_x_ptr[col];
			uchar y_value = abs_confidence_measure_y_ptr[col];
			
			confidence_map_ptr[col] = - sqrt(pow(x_value, 2) + pow(y_value, 2));
		}
	}
}

void var_measure
(
	InputArray disparity_map, 
	int r, 
	OutputArray confidence_map
){
	int win = 2 * r + 1;

	confidence_map.create(disparity_map.size(), CV_32F);

	Mat _disparity_map = disparity_map.getMat(),
		_confidence_map = confidence_map.getMat(),
		_mean, _pow, _diff, _var;

	blur(_disparity_map, _mean, Size(win, win));
	_diff = _disparity_map - _mean;
	multiply(_diff, _diff, _pow);
	blur(_pow, _var, Size(win, win));
	_confidence_map = -_var;
}

void disparity_ambiguity_measure
(
	InputArray c1_idx, 
	InputArray c2_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
	    _c2_idx = c2_idx.getMat(),
        _confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);
		uchar* c2_idx_ptr = _c2_idx.ptr<uchar>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			uchar d_1 = c1_idx_ptr[col]; 
			uchar d_2 = c2_idx_ptr[col];

			confidence_map_ptr[col] = - (float)abs(d_1 - d_2);
		}
	}
}

//2.4	The Entire Cost Curve
void winner_margin
(
	InputArray c1, 
	InputArray c2m, 
	InputArray c_sum, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1 = c1.getMat(), 
	    _c2m = c2m.getMat(), 
	    _c_sum = c2m.getMat(), 
    	_confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		float* c1_ptr = _c1.ptr<float>(row);
		float* c2m_ptr = _c2m.ptr<float>(row);
		float* c_sum_ptr = _c_sum.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			float c_1 = c1_ptr[col];
			float c_2m = c2m_ptr[col];
			float c_sum = c_sum_ptr[col];

			if (c_sum > 0)confidence_map_ptr[col] = (c_2m - c_1) / c_sum;
			else confidence_map_ptr[col] = 0;
		}
	}
}

void maximum_likelihood_measure
(
	InputArrayOfArrays costs, 
	InputArray c1, 
	float sigma, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _sum, 
		_c1 = c1.getMat(),
		_confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			float den = 0;

			for (int d = 0; d < _costs.size(); d++)
			{
				float value = _costs[d].ptr<float>(row)[col];

				den += exp(-value / (2 * sigma * sigma));
			}

			float c_1 = _c1.ptr<float>(row)[col];

			if (den != 0)
				_confidence_map.ptr<float>(row)[col] = exp(-c_1 / 2 * sigma * sigma) / den;
			else
				_confidence_map.ptr<float>(row)[col] = 0; 
		}
	}
}

void attainable_maximum_likelihood
(
	InputArrayOfArrays costs, 
	InputArray c1, 
	float sigma, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1 = c1.getMat(),
		_confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			float c_1 = _c1.ptr<float>(row)[col];
			float den = 0;

			for (int d = 0; d < _costs.size(); d++)
			{
				float value = _costs[d].ptr<float>(row)[col];

				den += exp(-(pow(value - c_1,2)) / (2 * sigma * sigma));
			}

			if (den != 0)
				_confidence_map.ptr<float>(row)[col] = 1 / den;
			else
				_confidence_map.ptr<float>(row)[col] = 0;
		}
	}
}

void negative_entropy_measure
(
	InputArrayOfArrays costs, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			float negative_entropy = 0;
			float den = 0;

			for (int d = 0; d < _costs.size(); d++)
			{
				float value = _costs[d].ptr<float>(row)[col];

				den += exp(-value); 
			}
			
			for (int d = 0; d < _costs.size(); d++)
			{
				float value = _costs[d].ptr<float>(row)[col];

				float p_d = exp(-value) / den;

				negative_entropy += (p_d * log2(p_d)); 	
			}

			_confidence_map.ptr<float>(row)[col] = negative_entropy;
		}
	}
}

void perturbation_measure
(
	InputArrayOfArrays costs, 
	InputArray c1, 
	InputArray c1_idx, 
	float s, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(), 
		_c1 = c1.getMat(),
		_confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);

	for (int row = 0; row < height; row++)
	{
		float* c1_ptr = _c1.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			float sum = 0;
			float c_1 = c1_ptr[col];
			uchar d_1 = c1_idx_ptr[col];
			
			for (int d = 0; d < _costs.size(); d++)
			{
				float c_d = _costs[d].ptr<float>(row)[col];
				float diff = (c_1 - c_d);

				if (d != d_1)
					sum += exp(- diff * diff) / (s * s);
			}

			_confidence_map.ptr<float>(row)[col] = - sum;
		}
	}
}

void local_minima_in_neighborhood
(
	InputArrayOfArrays local_minima, 
	InputArray c1_idx, 
	int r, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
		_confidence_map = confidence_map.getMat();

	vector<Mat> _local_minima; local_minima.getMatVector(_local_minima);

	for (int row = 0; row < height; row++)
	{
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			uchar c_1_idx = c1_idx_ptr[col];
			float sum = 0;

			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					if (row + i >= 0 && row + i < height && col + j >= 0 && col + j < width)
					{
						sum += _local_minima[c_1_idx].ptr<uchar>(row + i)[col + j];
					}
				}
			}

			confidence_map_ptr[col] = sum;
		}
	}
}

//2.5	Consistency between the left and right disparity maps
void left_right_consistency_check
(
	InputArray disparity_L2R, 
	InputArray disparity_R2L, 
	int disp_scale, 
	int bad, 
	int height, 
	int width, 
	OutputArray confidence_map
){

	confidence_map.create(height, width, CV_32F);

	Mat _left = disparity_L2R.getMat(),
	    _right = disparity_R2L.getMat(),
	    _confidence_map = confidence_map.getMat();

	_confidence_map.setTo(Scalar(0));

	for (int row = 0; row < height; row++)
	{
		float* left_ptr = _left.ptr<float>(row);
		float* right_ptr = _right.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			if (left_ptr[col] >= 0)
			{
				int j_right = col - (left_ptr[col]) / disp_scale;

				if (j_right < 0)
					confidence_map_ptr[col] = 0;
				else
				{
					if (right_ptr[j_right] >= 0)
					{
						int jRefp = j_right + (right_ptr[j_right]) / disp_scale;

						if((float)abs(col - jRefp) > bad)
						    confidence_map_ptr[col] = 0;
						else
					        confidence_map_ptr[col] = 255;
					}
				}
			}
		}
	}
}

void left_right_difference
(
	InputArray c1_L, 
	InputArray c2_L, 
	InputArray c1_R, 
	InputArray disparity_L, 
	int disp_scale, 
	float epsilon, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_L = c1_L.getMat(),
		_c2_L = c2_L.getMat(), 
	    _c1_R = c1_R.getMat(), 
	    _disparity_L = disparity_L.getMat(),
        _confidence_map = confidence_map.getMat();

	_confidence_map.setTo(Scalar(0));

	for (int row = 0; row < height; row++)
	{
		float* c1_L_ptr = _c1_L.ptr<float>(row);
		float* c2_L_ptr = _c2_L.ptr<float>(row);
		float* disparity_L_ptr = _disparity_L.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			float c1_l = c1_L_ptr[col]; 
			float c2_l = c2_L_ptr[col];
			int d1_l = (int)disparity_L_ptr[col]/disp_scale;
			
			if (col - d1_l >= 0 && d1_l>= 0)
			{
				float den = (float)abs(c1_l - _c1_R.ptr<float>(row)[col - d1_l]); 

				confidence_map_ptr[col] = ((c2_l - c1_l) + epsilon) / (den + epsilon);
			}		
		}
	}
}

void uniqueness_constraint
(
	InputArray c1, 
	InputArray c1_idx, 
	int height, 
	int width, 
	OutputArray confidence_map_1, 
	OutputArray confidence_map_2, 
	OutputArray confidence_map_3
){
	confidence_map_1.create(height, width, CV_32F);
	confidence_map_2.create(height, width, CV_32F);
	confidence_map_3.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
		_c1 = c1.getMat(),
		_confidence_map_1 = confidence_map_1.getMat(),
		_confidence_map_2 = confidence_map_2.getMat(),
		_confidence_map_3 = confidence_map_3.getMat(),
		_occurrence = Mat(height, width, CV_32F),
		_cost_register = Mat(1, width, CV_32F),
		_index_register = Mat(1, width, CV_32S);

	_confidence_map_1.setTo(Scalar(0));
	_occurrence.setTo(Scalar(0));
	_confidence_map_3.setTo(Scalar(-1));

	for (int row = 0; row < height; row++)
	{
		_cost_register.setTo(Scalar(-1));
		_index_register.setTo(Scalar(-1));

		float* c1_ptr = _c1.ptr<float>(row);
		float* cost_register_ptr = _cost_register.ptr<float>(0);
		int* index_register_ptr = _index_register.ptr<int>(0);
		float* confidence_map_1_ptr = _confidence_map_1.ptr<float>(row);
		float* occurrence_ptr = _occurrence.ptr<float>(row);
		float* confidence_map_3_ptr = _confidence_map_3.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);
		
		for (int col = 0; col < width; col++)
		{
			int index_r = col - c1_idx_ptr[col]; 

			if (index_r >= 0)
			{
				float cost = c1_ptr[col]; 
				float best_cost = cost_register_ptr[index_r]; 

				if (best_cost >= 0)
				{
					occurrence_ptr[index_r]++;

					if (cost < best_cost) 
					{
						cost_register_ptr[index_r] = cost; 
						int old_index = index_register_ptr[index_r]; 
						confidence_map_1_ptr[old_index] = 0;
						confidence_map_1_ptr[col] = 1;
						confidence_map_3_ptr[col] = - cost;
						index_register_ptr[index_r] = col; 
					}
					else
					{
						confidence_map_1_ptr[col] = 0;		
					}
				}
				else
				{
					cost_register_ptr[index_r] = cost; 
					index_register_ptr[index_r] = col; 
					confidence_map_1_ptr[col] = 1;
					occurrence_ptr[index_r]++;
					confidence_map_3_ptr[col] = - cost;
				}
			}
		}
	}

	for (int row = 0; row < height; row++)
	{
		float* occurrence_ptr = _occurrence.ptr<float>(row);
		float* confidence_map_2_ptr = _confidence_map_2.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			int index_r = col - c1_idx_ptr[col];
			confidence_map_2_ptr[col] = - occurrence_ptr[index_r];
		}
	}

	_confidence_map_1*=255;
}

void asymmetric_consistency_check
(
	InputArray c1, 
	InputArray c1_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
		_c1 = c1.getMat(),
		_confidence_map = confidence_map.getMat(),
		_cost_register = Mat(1, width, CV_32F),
		_disparity_register = Mat(1, width, CV_32S),
		_index_register = Mat(1, width, CV_32S);

	_confidence_map.setTo(Scalar(0));

	for (int row = 0; row < height; row++)
	{
		_cost_register.setTo(Scalar(-1));
		_disparity_register.setTo(Scalar(-1));
		_index_register.setTo(Scalar(-1));

		float* c1_ptr = _c1.ptr<float>(row);
		float* cost_register_ptr = _cost_register.ptr<float>(0);
		int* disparity_register_ptr = _disparity_register.ptr<int>(0);
		int* index_register_ptr = _index_register.ptr<int>(0);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);
		
		for (int col = 0; col < width; col++)
		{
			int index_r = col - c1_idx_ptr[col];
			int disp = c1_idx_ptr[col];

			if (index_r >= 0)
			{
				float cost = c1_ptr[col];
				float best_cost = cost_register_ptr[index_r];
				int max_disp = disparity_register_ptr[index_r];

				if (max_disp >= 0)
				{
					if (disp > max_disp)
					{
						disparity_register_ptr[index_r] = disp;
						int old_index = index_register_ptr[index_r];
						confidence_map_ptr[old_index] = 0;
						confidence_map_ptr[col] = 1;
						index_register_ptr[index_r] = col;

						if (cost >= best_cost)
						{
							confidence_map_ptr[col] = 0;
						}
						else
						{
							cost_register_ptr[index_r] = cost;
						}
					}
				}
				else
				{
					cost_register_ptr[index_r] = cost;
					disparity_register_ptr[index_r] = disp;
					index_register_ptr[index_r] = col;
					confidence_map_ptr[col] = 1;
				}
			}
		}
	}
}

//2.6	Matching cost between left and right image intensities
void horizontal_gradient
(
	InputArray left_stereo, 
	OutputArray confidence_map
)
{
	confidence_map.create(left_stereo.size(), CV_32F);
	Mat _confidence_map = confidence_map.getMat(),
        _left_stereo_gray, 
		_grad_x, 
		_grad_x_abs;

	//convert the image to grayscale
	if (left_stereo.getMat().channels() > 1)
		cvtColor(left_stereo, _left_stereo_gray, CV_BGR2GRAY);
	else
		_left_stereo_gray = left_stereo.getMat().clone();
	
	Sobel(_left_stereo_gray, _grad_x, left_stereo.depth(), 1, 0, 1, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(_grad_x, _grad_x_abs);
	_grad_x_abs.convertTo(_confidence_map, CV_32F);
}

void zero_mean_sum_of_absolute_differences
(
	InputArray left, 
	InputArray right, 
	InputArray c1, 
	InputArray c1_idx, 
	int r, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _left, _right,
	    _c1 = c1.getMat(), _c1_idx = c1_idx.getMat(),
		_abs_diff = Mat(left.size(), CV_32F),
		_abs_box_diff = Mat(left.size(), CV_32F),
		_mean_L, _mean_R,
		_confidence_map = confidence_map.getMat();

	left.getMat().convertTo(_left, CV_32F);
	right.getMat().convertTo(_right, CV_32F);

	_mean_L.create(left.size(), CV_32F);
	_mean_R.create(right.size(), CV_32F);

	for (int row = 0; row < _left.rows; row++)
	{
		float* left_ptr = _left.ptr<float>(row);
		float* right_ptr = _right.ptr<float>(row);
		float* abs_diff_ptr = _abs_diff.ptr<float>(row);
		uchar* index_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < _left.cols; col++)
		{
			uchar index = index_ptr[col];
			int col_R = (col - index < 0) ? 0 : col - index;
			abs_diff_ptr[col] = abs(left_ptr[col] - right_ptr[col_R]);
		}
	}

	box_filter(_left, r, _mean_L);
	box_filter(_right, r, _mean_R);
	box_filter(_abs_diff, r, _abs_box_diff);
	
	_mean_L = _mean_L / ((2 * r + 1) * (2 * r + 1));
	_mean_R = _mean_R / ((2 * r + 1) * (2 * r + 1));

	for (int row = 0; row < _left.rows; row++)
	{
		float* mean_L_ptr = _mean_L.ptr<float>(row);
		float* mean_R_ptr = _mean_R.ptr<float>(row);
		float* abs_box_diff_ptr = _abs_box_diff.ptr<float>(row);
		float* left_ptr = _left.ptr<float>(row);
		float* right_ptr = _right.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* index_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < _left.cols; col++)
		{
			uchar index = index_ptr[col];
			int col_R = (col - index < 0) ? 0 : col - index;
			confidence_map_ptr[col] = - abs(abs_box_diff_ptr[col] - mean_L_ptr[col] + mean_R_ptr[col_R]);
		}
	}
}

//2.7	Distinctiveness-based confidence measures
void distinctiveness
(
	InputArrayOfArrays costs_LL, 
	int d_min, 
	int d_max, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	vector<Mat> _costs_LL;
	costs_LL.getMatVector(_costs_LL);

	Mat min = confidence_map.getMat();

	min.setTo(Scalar(-1));

	float minimum;

	for (int row = 0; row < height; row++)
	{
		float* min_ptr = min.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			minimum = FLT_MAX;

			for (int d = 0; d < _costs_LL.size(); d++)
			{
				float value = _costs_LL[d].ptr<float>(row)[col];

				if (value < minimum && value >= 0 && d != - d_min)
				{
					minimum = value;
				}
			}

			min_ptr[col] = minimum;
		}
	}
}

void distinctive_similarity_measure
(
	InputArray distinctiveness_L, 
	InputArray distinctiveness_R, 
	InputArray c1, 
	InputArray c1_idx, 
	float epsilon, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _distinctiveness_L = distinctiveness_L.getMat(),
		_distinctiveness_R = distinctiveness_R.getMat();

	Mat _c1 = c1.getMat(), 
        _c1_idx = c1_idx.getMat(),
        _confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		float* c1_ptr = _c1.ptr<float>(row);
		float* distinctiveness_L_ptr = _distinctiveness_L.ptr<float>(row);
		float* distinctiveness_R_ptr = _distinctiveness_R.ptr<float>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* c1_idx_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			uchar d_1 = c1_idx_ptr[col]; 
			uchar index = col - d_1;

			if (index >= 0)
			{
				float d_l = distinctiveness_L_ptr[col];
				float d_r = distinctiveness_R_ptr[index];
				float c_1 = c1_ptr[col];

				confidence_map_ptr[col] = ((d_l * d_r) + epsilon) / ((c_1 * c_1) + epsilon);
			}
		}
	}
}

void self_aware_matching
(
	InputArrayOfArrays costs, 
	InputArrayOfArrays costs_LL, 
	InputArray c1_idx, 
	int d_min, 
	int d_max, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _c1_idx = c1_idx.getMat(),
	    _confidence_map = confidence_map.getMat();

	vector<Mat> _costs; costs.getMatVector(_costs);
	vector<Mat> _costs_LL; costs_LL.getMatVector(_costs_LL);

	float sum, 
		  sum_mean_LR, sum_mean_LL, 
		  sum_deviation_LR, sum_deviation_LL,
          value_LR, value_LL,
		  mean_LR, mean_LL,
		  standard_deviation_LR, standard_deviation_LL;

	int count_LR, count_LL, 
		num_disp = d_max - d_min + 1;

	for (int row = 0; row < height; row++)
	{
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);
		uchar* _c1_idx_ptr = _c1_idx.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			count_LR = 0, count_LL = 0;
			sum_mean_LR = 0, sum_mean_LL = 0;
			uchar d1 = _c1_idx_ptr[col]; 
			int offset = - d_min - d1; 

			//mean computation
			for (int d = 0; d < _costs.size(); d++)
			{
				if (d + offset < _costs_LL.size() && d + offset >= 0)
				{
					value_LR = _costs[d].ptr<float>(row)[col];
					sum_mean_LR += value_LR;
					count_LR++;

					value_LL = _costs_LL[d + offset].ptr<float>(row)[col];
					sum_mean_LL += value_LL;
					count_LL++;
				}
			}

			mean_LR = sum_mean_LR / count_LR;
			mean_LL = sum_mean_LL / count_LL;

			//standard deviation and SAMM computation
			sum = 0, sum_deviation_LR = 0, sum_deviation_LL = 0;

			for (int d = 0; d < _costs.size(); d++)
			{
				if (d + offset < _costs_LL.size() && d + offset >= 0)
				{
					value_LR = _costs[d].ptr<float>(row)[col];
					sum_deviation_LR += pow(value_LR - mean_LR, 2);

					value_LL = _costs_LL[d + offset].ptr<float>(row)[col];
					sum_deviation_LL += pow(value_LL - mean_LL, 2);

					sum += (value_LR - mean_LR)*(value_LL - mean_LL);
				}
			}

			standard_deviation_LR = sqrt(sum_deviation_LR);
			standard_deviation_LL = sqrt(sum_deviation_LL);

			confidence_map_ptr[col] = sum / sqrt(sum_deviation_LR * sum_deviation_LL);
		}
	}
}

//2.8  Based on image information
void distance_to_border
(
	InputArray image, 
	OutputArray confidence_map
){
	Mat _image = image.getMat();

	int height = _image.rows;
	int width = _image.cols;

	Mat _border_map = Mat(height, width, CV_8U);
	_border_map.setTo(Scalar(1));

	confidence_map.create(_image.size(), CV_32F);
	Mat _confidence_map = confidence_map.getMat();
	
	for (int row = 0; row < height; row++)
	{
		uchar* border_map_ptr = _border_map.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			if (row == 0) border_map_ptr[col] = 0;
			if (col == 0) border_map_ptr[col] = 0;
			if (row == height - 1) border_map_ptr[col] = 0;
			if (col == width  - 1) border_map_ptr[col] = 0;
		}
	}

	distanceTransform(_border_map, _confidence_map, CV_DIST_L2, 3);
}

void distance_to_left_border
(
	InputArray image, 
	int d_max, 
	OutputArray confidence_map
){
	Mat _image = image.getMat();

	int height = _image.rows;
	int width = _image.cols;

	Mat _border_map = Mat(height, width, CV_8U);
	_border_map.setTo(Scalar(1));

	confidence_map.create(_image.size(), CV_32F);
	Mat _confidence_map = confidence_map.getMat();

	for (int row = 0; row < height; row++)
	{
		uchar* border_map_ptr = _border_map.ptr<uchar>(row);

		for (int col = 0; col < width; col++)
		{
			if (col == 0) border_map_ptr[col] = 0;

			if(col < d_max)
				_confidence_map.ptr<float>(row)[col] = 0;
			else
				_confidence_map.ptr<float>(row)[col] = 255;
		}
	}
}

void compute_DD
(
	InputArray image, 
	int lowThreshold, 
	int ratio, 
	int r, 
	OutputArray confidence_map
){
	Mat _image_gray, _image, _edges, _dst;;

	image.getMat().convertTo(_image, CV_8U);
	_dst.create(_image.size(), _image.depth());

	if (_image.channels() > 1)
		cvtColor(_image, _image_gray, CV_BGR2GRAY);
	else
		_image_gray = _image.clone();

	blur(_image_gray, _edges, Size(3, 3));
	Canny(_edges, _edges, lowThreshold, lowThreshold * ratio, 2 * r + 1);

	_dst.setTo(Scalar(0));
	_image.copyTo(_dst, _edges);

	threshold(_dst, _dst, 0, 255, 1);
	distanceTransform(_dst, confidence_map, CV_DIST_L2, 3);
}

//2.9 Based on disparity map
void difference_with_median
(
	InputArray disparity_L2R, 
	int r, 
	int bad, 
	int height, 
	int width, 
	OutputArray confidence_map
){
	confidence_map.create(height, width, CV_32F);

	Mat _disparity_L2R = Mat(height, width, CV_8U),
		_confidence_map = confidence_map.getMat(),
		_median;

	disparity_L2R.getMat().convertTo(_disparity_L2R, CV_8U);

	//compute median
	medianBlur(_disparity_L2R, _median, 2 * r + 1);

	for (int row = 0; row < height; row++)
	{
		uchar* _disparity_L2R_ptr = _disparity_L2R.ptr<uchar>(row);
		uchar* median_ptr = _median.ptr<uchar>(row);
		float* confidence_map_ptr = _confidence_map.ptr<float>(row);

		for (int col = 0; col < width; col++)
		{
			if((float)abs(_disparity_L2R_ptr[col] - median_ptr[col]) > bad)
                confidence_map_ptr[col] = 0;
			else
				confidence_map_ptr[col] = 255;
		}
	}
}

void fn_confidence_measure
(
	InputArray image_L, 
	InputArray image_R, 
	_DSI dsi_LR, 
	_DSI dsi_LL, 
	_DSI dsi_RR, 
	int bad, 
	vector<string> choices, 
	OutputArray disparity_map, 
	vector<string> &methods, 
	OutputArrayOfArrays confidences
){
	//computation
	cout << string( 2, '\n' ) 
	     << "***** Compute Base Costs *****" 
	     << string( 2, '\n' );

	//confidence measures 
	vector<Mat> _confidences;

	//confidence measure params 
	confParams params;
	set_parameters(params);

	//base info
	int height = dsi_LR.height;
	int width = dsi_LR.width;
	int d_min = dsi_LR.d_min;
	int d_max = dsi_LR.d_max;
	int num_disp = dsi_LR.num_disp;
	int disp_scale = 256 / num_disp;
	vector<Mat> costs = dsi_LR.values;
	vector<Mat> costs_LL = dsi_LL.values;
	vector<Mat> costs_RR = dsi_RR.values;

	//generate right dsi.
	cout  << " - generate right dsi..." << endl;
	_DSI dsi_RL = DSI_left2right(dsi_LR);

	//compute left and right disparity maps
	cout  << " - compute left and right disparity maps..." << endl;
	Mat disparity_L2R = disparity_map_L2R(dsi_LR);
	Mat disparity_R2L = disparity_map_R2L(dsi_LR);

	//compute c_1, c_2, c^_2 in the paper as well as sum of matching costs and
	//number of inflection points(NOI).
	cout  << " - compute c_1, c_2, c^_2 in the paper as well as sum of matching costs..." << endl;
	vector<Mat> local_minima;
	Mat c1, c1_idx, c2, c2_idx, c_hat_2, c_hat_2_idx, c_sum, NOI, c1_R; 
	compute_base_costs(dsi_LR.values, c1, c1_idx, c2, c2_idx, c_hat_2, c_hat_2_idx, c_sum, NOI, local_minima);

	//compute minimum right to left
	cout << " - compute minimum right to left..." << endl;
	minimum(dsi_RL.values, c1_R);

	//computation
	cout << string( 2, '\n' ) 
     << "***** Confidence Measure  *****" 
     << string( 2, '\n' );

	// 01. Matching Score Measure (MSM).
	if (use(choices, "msm") == true)
	{
		Mat msm;
		cout  << " - confidence measure: matching score measure (MSM)" << endl; 
		matching_score_measure(c1, height, width, msm);
		_confidences.push_back(msm);
		methods.push_back("msm");
	}

	// 02. Local Curve (LC).
	if (use(choices, "lc") == true)
	{
		Mat lc;
		cout  << " - confidence measure: local curve (LC)" << endl;
		local_curve(costs, c1_idx, params.lc_gamma, height, width, lc);
		_confidences.push_back(lc);
		methods.push_back("lc");
	}

	// 03. Curvature (CUR).
	if (use(choices, "cur") == true)
	{
		Mat cur;
		cout  << " - confidence measure: curvature (CUR)" << endl;
		curvature(costs, c1_idx, height, width, cur);
		_confidences.push_back(cur);
		methods.push_back("cur");
	}

	// 04. Peak Ratio (PKR).
	if (use(choices, "pkr") == true)
	{
		Mat pkr;
		cout  << " - confidence measure: peak ratio (PKR)" << endl;
		peak_ratio(c1, c_hat_2, params.pkr_epsilon, height, width, pkr);
		_confidences.push_back(pkr);
		methods.push_back("pkr");
	}

	// 05. Peak Ratio Naive (PKRN).
	if (use(choices, "pkrn") == true)
	{
		Mat pkrn;
		cout  << " - confidence measure: peak ratio naive (PKRN)" << endl;
		peak_ratio(c1, c2, params.pkr_epsilon, height, width, pkrn);
		_confidences.push_back(pkrn);
		methods.push_back("pkrn");
	}

	// 06. Average Peak Ratio (APKR).
	if (use(choices, "apkr") == true)
	{
		Mat apkr;
		cout  << " - confidence measure: average peak ratio (APKR)" << endl;
		average_peak_ratio(costs, c1_idx, c_hat_2_idx, params.apkr_epsilon, params.apkr_radius, height, width, apkr);
		_confidences.push_back(apkr);
		methods.push_back("apkr");
	}

	// 07. Average Peak Ratio Naive (APKRN).
	if (use(choices, "apkrn") == true)
	{
		Mat apkrn;
		cout  << " - confidence measure: average peak ratio naive (APKRN)" << endl;
		average_peak_ratio(costs, c1_idx, c2_idx, params.apkr_epsilon, params.apkr_radius, height, width, apkrn);
		_confidences.push_back(apkrn);
		methods.push_back("apkrn");
	}

	// 08. Maximum Margin (MM).
	if (use(choices, "mm") == true)
	{
		Mat mm;
		cout  << " - confidence measure: maximum margin (MM)" << endl;
		maximum_margin(c1, c_hat_2, height, width, mm);
		_confidences.push_back(mm);
		methods.push_back("mm");
	}

	// 09. Maximum Margin Naive (MMN).
	if (use(choices, "mmn") == true)
	{
		Mat mmn;
		cout  << " - confidence measure: maximum margin naive (MMN)" << endl;
		maximum_margin(c1, c2, height, width, mmn);
		_confidences.push_back(mmn);
		methods.push_back("mmn");
	}

	// 10. Nonlinear Maximum Margin
	if (use(choices, "nmm") == true)
	{
		Mat nmm;
		cout  << " - confidence measure: nonlinear margin (NMM)" << endl;
		non_linear_margin(c1, c_hat_2, params.nmm_sigma, height, width, nmm);
		_confidences.push_back(nmm);
		methods.push_back("nmm");
	}

	// 11. Nonlinear Maximum Margin Naive
	if (use(choices, "nmmn") == true)
	{
		Mat nmmn;
		cout  << " - confidence measure: nonlinear margin naive (NMMN)" << endl;
		non_linear_margin(c1, c2, params.nmm_sigma, height, width, nmmn);
		_confidences.push_back(nmmn);
		methods.push_back("nmmn");
	}

	// 12. Disparity Variance Measure (DVM).
	if (use(choices, "dvm") == true)
	{
		Mat dvm;
		cout  << " - confidence measure: disparity variance measure (DVM)" << endl;

		disparity_variance_measure(disparity_L2R, params.dvm, dvm);
		_confidences.push_back(dvm);
		methods.push_back("dvm");
	}

	// 13. Disparity Ambiguity Measure (DAM).
	if (use(choices, "dam") == true)
	{
		Mat dam;
		cout  << " - confidence measure: disparity ambiguity measure (DAM)" << endl;
		disparity_ambiguity_measure(c1_idx, c2_idx, height, width, dam);
		_confidences.push_back(dam);
		methods.push_back("dam");
	}

	// 14. Maximum Likelihood Measure (MLM).
	if (use(choices, "mlm") == true)
	{
		Mat mlm;
		cout  << " - confidence measure: maximum likelihood measure (MLM)" << endl;
		maximum_likelihood_measure(costs, c1, params.mlm_sigma, height, width, mlm);
		_confidences.push_back(mlm);
		methods.push_back("mlm");
	}

	// 15. Attainable Maximum Likelihood (AML).
	if (use(choices, "aml") == true)
	{
		Mat aml;
		cout  << " - confidence measure: attainable maximum likelihood (AML)" << endl;
		attainable_maximum_likelihood(costs, c1, params.aml_sigma, height, width, aml);
		_confidences.push_back(aml);
		methods.push_back("aml");
	}

	// 16. Negative Entropy Measure (NEM).
	if (use(choices, "nem") == true)
	{
		Mat nem;
		cout  << " - confidence measure: negative entropy (NEM)" << endl;
		negative_entropy_measure(costs, height, width, nem);
		_confidences.push_back(nem);
		methods.push_back("nem");
	}

	// 17. Number Of Inflections (NOI).
	if (use(choices, "noi") == true)
	{
		cout  << " - confidence measure: number of inflections (NOI)" << endl;
		_confidences.push_back(NOI);
		methods.push_back("noi");
	}

	// 18. Winner Margin (WMN).
	if (use(choices, "wmn") == true)
	{
		Mat wmn;
		cout  << " - confidence measure: winner margin (WMN)" << endl;
		winner_margin(c1, c_hat_2, c_sum, height, width, wmn);
		_confidences.push_back(wmn);
		methods.push_back("wmn");
	}

	// 19. Winner Margin Naive (WMNN).
	if (use(choices, "wmnn") == true)
	{
		Mat wmnn;
		cout  << " - confidence measure: winner margin naive (WMNN)" << endl;
		winner_margin(c1, c2, c_sum, height, width, wmnn);
		_confidences.push_back(wmnn);
		methods.push_back("wmnn");
	}

	// 20. Perturbation Measure (PER).
	if (use(choices, "per") == true)
	{
		Mat per;
		cout  << " - confidence measure: perturbation measure (PER)" << endl;
		perturbation_measure(costs, c1, c1_idx, params.per_sigma, height, width, per);
		_confidences.push_back(per);
		methods.push_back("per");
	}

	// 21. Left Right Consistency Check (LRC).
	if (use(choices, "lrc") == true)
	{
		Mat lrc;
		cout  << " - confidence measure: left right consistency check (LRC)" << endl;
		left_right_consistency_check(disparity_L2R, disparity_R2L, disp_scale, bad, height, width, lrc);
		_confidences.push_back(lrc);
		methods.push_back("lrc");
	}

	// 22. Asymmetric Consistency Check (ACC).
	if (use(choices, "acc") == true)
	{
		Mat acc;
		cout  << " - confidence measure: asymmetric consistency check (ACC)" << endl;
		asymmetric_consistency_check(c1, c1_idx, height, width, acc);
		_confidences.push_back(acc);
		methods.push_back("acc");
	}

	// 23. Uniqueness Constraint (UC).
	if (use(choices, "uc") == true)
	{
		Mat uc_binary, uc_occurence, uc_cost;
		cout  << " - confidence measure: uniqueness constraint (UC)" << endl;
		uniqueness_constraint(c1, c1_idx, height, width, uc_binary, uc_occurence, uc_cost);
		_confidences.push_back(uc_binary);

		methods.push_back("uc");
		//methods.push_back("uc_occurence");
		//methods.push_back("uc_cost");
	}

	// 24. Left Right Difference (LRD).
	if (use(choices, "lrd") == true)
	{
		Mat lrd;
		cout  << " - confidence measure: left right difference (LRD)" << endl;
		left_right_difference(c1, c2, c1_R, disparity_L2R, disp_scale, params.lrd_epsilon, height, width, lrd);
		_confidences.push_back(lrd);
		methods.push_back("lrd");
	}

	// 25. Horizontal Gradient Measure (HGM).
	if (use(choices, "hgm") == true)
	{
		Mat hgm;
		cout  << " - confidence measure: horizontal gradient (HGM)" << endl;
		horizontal_gradient(image_L, hgm);
		_confidences.push_back(hgm);
		methods.push_back("hgm");
	}

	// 26. Zero Mean Sum Of Absolute Difference (ZSAD).
	if (use(choices, "zsad") == true)
	{
		Mat zsad;
		cout  << " - confidence measure: zero mean sum of absolute difference (ZSAD)" << endl;
		zero_mean_sum_of_absolute_differences(image_L, image_R, c1, c1_idx, params.zsad_radius, height, width, zsad);
		_confidences.push_back(zsad);
		methods.push_back("zsad");
	}

	// 27. Distinctiveness (DTS).
	if (use(choices, "dts") == true)
	{
		Mat dts_L;
		cout  << " - confidence measure: distinctiveness (DTS)" << endl;
		distinctiveness(costs_LL, dsi_LL.d_min, dsi_LL.d_max, height, width, dts_L);
		_confidences.push_back(dts_L);
		methods.push_back("dts");
	}

	// 28. Distinctive Similarity Measure (DSM).
	if (use(choices, "dsm") == true)
	{
		Mat dsm, distinctiveness_R, distinctiveness_L;
		cout  << " - confidence measure: distinctive similarity measure (DSM)" << endl;

		distinctiveness(costs_LL, dsi_LL.d_min, dsi_LL.d_max, height, width, distinctiveness_L);
		distinctiveness(costs_RR, dsi_RR.d_min, dsi_RR.d_max, height, width, distinctiveness_R);

		distinctive_similarity_measure(distinctiveness_L, distinctiveness_R, c1, c1_idx, params.dsm_epsilon, height, width, dsm);
		_confidences.push_back(dsm);
		methods.push_back("dsm");
	}

	// 29. Self Aware Matching Measure(SAMM).
	if (use(choices, "samm") == true)
	{
		Mat samm;
		cout  << " - confidence measure: self aware matching measure (SAMM)" << endl;
		self_aware_matching(dsi_LR.values, dsi_LL.values, c1_idx, dsi_LL.d_min, dsi_LL.d_max, height, width, samm);
		_confidences.push_back(samm);
		methods.push_back("samm");
	}

	// 30. Distance To Border (DB).
	if (use(choices, "db") == true)
	{
		Mat db;
		cout  << " - confidence measure: distance to border (DB)" << endl;
		distance_to_border(disparity_L2R, db);
		_confidences.push_back(db);
		methods.push_back("db");
	}

	// 31. Distance To Border Left (DBL).
	if (use(choices, "dbl") == true)
	{
		Mat dbl;
		cout  << " - confidence measure: distance to left border (DBL)" << endl;
		distance_to_left_border(disparity_L2R, dsi_LL.d_max, dbl);
		_confidences.push_back(dbl);
		methods.push_back("dbl");
	}

	// 32. Distance To Discontinuity (DTD).
	if (use(choices, "dtd") == true)
	{
		Mat dtd;
		cout  << " - confidence measure: distance to discontinuity (DTD)" << endl;
		compute_DD(disparity_L2R, params.DD_max_lowThreshold, params.DD_ratio, params.DD_radius, dtd);
		_confidences.push_back(dtd);
		methods.push_back("dtd");
	}

	// 33. Distance To Edge (DTE).
	if (use(choices, "dte") == true)
	{
		Mat dte;
		cout  << " - confidence measure: distance to edge (DTE)" << endl;
		compute_DD(image_L, params.DD_max_lowThreshold, params.DD_ratio, params.DD_radius, dte);
		_confidences.push_back(dte);
		methods.push_back("dte");
	}

	// 34. Difference With Median (MED).
	if (use(choices, "med") == true)
	{
		Mat med;
		cout  << " - confidence measure: difference with median (MED)" << endl;
		difference_with_median(disparity_L2R, params.med_radius, bad, height, width, med);
		_confidences.push_back(med);
		methods.push_back("med");
	}

	// 35. Local Minima In Neighborhood (LMN).
	if (use(choices, "lmn") == true)
	{
		Mat lmn;
		cout  << " - confidence measure: local minima in neighborhood (LMN)" << endl;
		local_minima_in_neighborhood(local_minima, c1_idx, params.lmn_radius, height, width, lmn);
		_confidences.push_back(lmn);
		methods.push_back("lmn");
	}

	//confidence maps
	confidences.create(_confidences.size(), 1, Mat(_confidences).type());

	for (int i = 0; i < _confidences.size(); i++)
		confidences.getMatRef(i) = _confidences[i];
	
	//disparity map 
	disparity_map.getMatRef() = disparity_L2R;
}
