#include "generate_samples.hpp"

using namespace cv;
using namespace std;

/***************************************************************************************************/
/*
* Utility function to distinguish binary confidence measures (0/255) 
* from non-binary confidence measures
*
/***************************************************************************************************/
bool is_binary
(
	InputArray confidence
){
	Mat _confidence = confidence.getMat();

	for (int row =0; row < _confidence.rows; row++)
	{
		for(int col =0; col < _confidence.cols; col++)
	 	{
			if(!(_confidence.ptr<float>(row)[col]== 0 || 
			    _confidence.ptr<float>(row)[col]== 255))
			return false;
		}
	}

	return true;
}


void split
(
	InputArray confidence, 
	float threshold0, float threshold1,
	OutputArray positive_samples, 
	OutputArray negative_samples
)
{
	Mat _confidence = confidence.getMat();
	positive_samples.create(_confidence.rows, _confidence.cols, CV_32F);
	negative_samples.create(_confidence.rows, _confidence.cols, CV_32F);

	Mat _positive = positive_samples.getMat();
	Mat _negative = negative_samples.getMat();

	_positive.setTo(Scalar(0));
	_negative.setTo(Scalar(0));

	int count = 0, 
	    size = _confidence.rows * _confidence.cols;

	float *confidence_values = (float*)malloc(sizeof(float)*size);
	float *rows = (float*)malloc(sizeof(float)*size);
	float *cols = (float*)malloc(sizeof(float)*size);	

	for (int row =0; row < _confidence.rows; row++)
	{
		for(int col =0; col < _confidence.cols; col++)
	 	{
			confidence_values[count] = _confidence.ptr<float>(row)[col];
			rows[count] = (float)row;
			cols[count] = (float)col;
			count++;
		}
	}

	int p = 0, r = size - 1;
	MergeSort(confidence_values,rows, cols, p, r, size); 
	reverse(confidence_values, rows, cols,  size);

	for (int i =0; i <= (int)size * threshold1; i++)
	{
		int row = (int)rows[i];
		int col = (int)cols[i];

		_positive.ptr<float>(row)[col] = 255;
	}

	for (int i = size; i >= (int)(size * (1 - threshold0)); i--)
	{
		int row = (int)rows[i];
		int col = (int)cols[i];

		_negative.ptr<float>(row)[col] = 255;
	}

	free(confidence_values);
	free(rows);
}

void generate_training_samples
(
	InputArrayOfArrays confidence_measures, 
	InputArray disparity_map, 
	float threshold0, float threshold1,
	vector<string> confidence_names, 
	vector<string> choices_positive, 
	vector<string> choices_negative, 
	OutputArray correct, OutputArray wrong
){
	vector<Mat> _confidence_measures;
	confidence_measures.getMatVector(_confidence_measures);

	Mat _disparity_map = disparity_map.getMat();

	correct.create(_disparity_map.rows, _disparity_map.cols, CV_32F);
	wrong.create(_disparity_map.rows, _disparity_map.cols, CV_32F);
	Mat _correct = correct.getMat(), _wrong = wrong.getMat();

	vector<Mat> _positive_training_samples, _negative_training_samples;

	for(int n = 0; n < _confidence_measures.size(); n++)
	{
		Mat _confidence = _confidence_measures.at(n),
		    _positive_samples, _negative_samples;

		if(!is_binary(_confidence))
		{
			split(_confidence, threshold0, threshold1, _positive_samples, _negative_samples);
		}
		else
		{
			_positive_samples = _confidence;
		    _negative_samples = 255 - _positive_samples;			
		}

		if(use(choices_positive, confidence_names.at(n)) == true)
		   _positive_training_samples.push_back(_positive_samples);

		if(use(choices_negative, confidence_names.at(n)) == true)   
		   _negative_training_samples.push_back(_negative_samples);
	}

	for(int row = 0; row < _disparity_map.rows; row++)
	{
		float* correct_ptr = _correct.ptr<float>(row);
		float* wrong_ptr = _wrong.ptr<float>(row);
		float* disparity_map_ptr = _disparity_map.ptr<float>(row);

		for(int col = 0; col < _disparity_map.cols; col++)
		{
			float P = 255, N = 255;

			for(int n = 0; n < _positive_training_samples.size(); n++)
			{
				if(!(_positive_training_samples.at(n).ptr<float>(row)[col] == 255 
					&& P == 255 && disparity_map_ptr[col]!=0))
					P = 0;
			}

			for(int n = 0; n < _negative_training_samples.size(); n++)
			{
				if(disparity_map_ptr[col]==0)
				{
					N = 255;
					break;	
				}
					
				if(!(_negative_training_samples.at(n).ptr<float>(row)[col] == 255 && N == 255))
					N = 0;
			}

			correct_ptr[col] = P;
			wrong_ptr[col] = N;
		}
	}
}
