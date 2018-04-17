#include "evaluation.hpp"

using namespace cv;
using namespace std;

int count_valid
(
	InputArray image, 
	float invalid
){
	Mat _image = image.getMat();
	int count = 0; 
	
	for(int row = 0; row < _image.rows; row++)
	{
		for(int col = 0; col < _image.cols; col++)
		{
			if(_image.ptr<float>(row)[col] != invalid)
			   count++;
		}
	}
	
	return count;
}

void samples_on_image
(
	InputArray left, 
	InputArray positive, 
	InputArray negative,
	OutputArray output
) 
{ 
	Mat _left = left.getMat(),
	    _positive = positive.getMat(),
	    _negative = negative.getMat();

    output.create(Size(_left.cols, _left.rows), CV_8UC3);
	vector<Mat> channels, channels_left; 
	split(output, channels); 
	Mat _output = output.getMat();

	if(_left.type() == CV_8UC3)
	   split(_left, channels_left);

	for(int row=0; row < _left.rows; row++) 
	{ 
		for(int col=0; col < _left.cols; col++) 
		{ 
			if(_positive.ptr<float>(row)[col] == 255) 
			{ 
				channels[0].ptr<uchar>(row)[col] = 0; 
				channels[1].ptr<uchar>(row)[col] = 255; 
				channels[2].ptr<uchar>(row)[col] = 0; 
			} 
			else if (_negative.ptr<float>(row)[col] == 255) 
			{ 
				channels[0].ptr<uchar>(row)[col] = 0; 
				channels[1].ptr<uchar>(row)[col] = 0; 
				channels[2].ptr<uchar>(row)[col] = 255; 
			} 
			else 
			{ 
				if(_left.type() == CV_8UC3)
				{
					channels[0].ptr<uchar>(row)[col] = channels_left[0].ptr<uchar>(row)[col]; 
					channels[1].ptr<uchar>(row)[col] = channels_left[1].ptr<uchar>(row)[col]; 
					channels[2].ptr<uchar>(row)[col] = channels_left[2].ptr<uchar>(row)[col];					
				}
				else
				{
					channels[0].ptr<uchar>(row)[col] = _left.ptr<uchar>(row)[col]; 
					channels[1].ptr<uchar>(row)[col] = _left.ptr<uchar>(row)[col]; 
					channels[2].ptr<uchar>(row)[col] = _left.ptr<uchar>(row)[col];
				}
			} 
		} 
	} 

	merge(channels, output);  
}

void statistics
(
	InputArray positive_samples, 
	InputArray negative_samples, 
	int& n_samples, 
	int& image_points, 
	int& n_positive_sample, 
	int & n_negative_sample, 
	float& density_samples, 
	float& density_positive, 
	float& density_negative
){
	Mat _positive_samples = positive_samples.getMat(),
	    _negative_samples = negative_samples.getMat();
	
	for(int row = 0; row < _positive_samples.rows; row++)
	{
		for(int col = 0; col < _positive_samples.cols; col++)
		{
			if(_positive_samples.ptr<float>(row)[col] == 255)
				n_positive_sample++;
		
			if(_negative_samples.ptr<float>(row)[col] == 255)
				n_negative_sample++;   
		}
	}

    image_points = _positive_samples.rows * _positive_samples.cols;
	n_samples = (float) n_positive_sample + (float) n_negative_sample;
	density_samples = (float)n_samples / (float) image_points;
	density_positive =  (float) n_positive_sample / (float) image_points;
	density_negative=  (float) n_negative_sample / (float) image_points;
}

void eval
(
	InputArray disparity, 
	InputArray groundtruth, 
	InputArray mask, 
	int threshold, 
	int invalid_disparity, 
	OutputArray error_map, 
	float &density, 
	float &err, 
	float &n_error, 
	float &tot
){
	error_map.create(groundtruth.size(), CV_32FC3);

	Mat _groundtruth = groundtruth.getMat(),
		_disparity = disparity.getMat(),
		_error_map = error_map.getMat();

	Mat _mask;

	if (mask.empty())
	{
		_mask = Mat(_groundtruth.size(), CV_32F);
		_mask.setTo(255);
	}
	else
	{
		_mask = mask.getMat();
	}

	int N_area = 0, total = 0;
	int error = 0;
	_error_map.setTo(Scalar(255));
	
	vector<Mat> channels;
	split(error_map, channels);
	
	channels.at(0).setTo(Scalar(0));
	channels.at(1).setTo(Scalar(0));
	channels.at(2).setTo(Scalar(0));

	for (int row = 0; row < _disparity.rows; row++)
	{
		float *groundtruth_ptr = _groundtruth.ptr<float>(row);
		float *disparity_ptr = _disparity.ptr<float>(row);
		float *mask_ptr = _mask.ptr<float>(row);
		float *error_map_ptr = _error_map.ptr<float>(row);

		for (int col = 0; col < _disparity.cols; col++)
		{
			total++;

			if (mask_ptr[col] == 255 && groundtruth_ptr[col] != invalid_disparity)
			{
				float disp_value = (float)disparity_ptr[col];
				float gt_value = (float)groundtruth_ptr[col];
				float difference = disp_value - gt_value;

				if ((float)fabs(difference) > threshold)
				{
					error++;
					channels.at(2).ptr<float>(row)[col] = 255;
				}
				else
				{
				    channels.at(1).ptr<float>(row)[col] = 255;
				}
				
				N_area++;
			}
			else if(groundtruth_ptr[col] == invalid_disparity)
			{
				channels.at(0).ptr<float>(row)[col] = 0;
				channels.at(1).ptr<float>(row)[col] = 0;
				channels.at(2).ptr<float>(row)[col] = 0;
			}
			else if (mask_ptr[col] == 128)
			{
				channels.at(0).ptr<float>(row)[col] = 128;
				channels.at(1).ptr<float>(row)[col] = 128;
				channels.at(2).ptr<float>(row)[col] = 128;
			}
		}
	}
	
	merge(channels, error_map);

    n_error = error;
	tot = N_area;
	err = (float)error / N_area * 100;
	density = (float)(N_area) / total * 100;
}

void eval_and_print
(
	InputArray disparity, 
	InputArray positive_samples, 
	InputArray negative_samples, 
	string gt_path, 
	float bad, 
	float invalid, 
	int scale, 
	string output_path,
	string file_output_path
){
	//eval
	float density, err, tot_positive, tot_negative, false_positive, true_negative;
	Mat groundtruth = imread(gt_path, CV_LOAD_IMAGE_UNCHANGED), error_map;
	groundtruth.convertTo(groundtruth, CV_32F);
	groundtruth = groundtruth / (float)scale;

	eval(disparity, groundtruth, positive_samples, bad, invalid, error_map, density, err, false_positive, tot_positive);
	eval(disparity, groundtruth, negative_samples, bad, invalid, error_map, density, err, true_negative, tot_negative);

	float true_positive = tot_positive - false_positive;
    float false_negative = tot_negative - true_negative;
	float accuracy = (float)(true_positive + true_negative) / (float)(true_positive + false_positive + true_negative + false_negative);
	float n_intersection_positive_with_gt = tot_positive;
	float n_intersection_negative_with_gt = tot_negative;
	float n_intersection_with_gt = n_intersection_positive_with_gt + n_intersection_negative_with_gt;
	float n_gt_points = count_valid(groundtruth, invalid);
    float density_samples = 0, density_positive = 0, density_negative = 0;
    int image_points = 0, n_samples = 0, n_positive_sample = 0, n_negative_sample = 0;

	statistics(positive_samples, negative_samples, n_samples, image_points,
			   n_positive_sample, n_negative_sample, density_samples, 
			   density_positive, density_negative);	
	//print
	string file_path = file_output_path;
	ifstream f(file_path.c_str());
    bool exist = f.good();
	ofstream file; 
	file.open(file_path.c_str(), fstream::in | fstream::out | fstream::app);

	if(!exist)
		file << "Name, n_image_points, n_samples, n_positive_sample,"
			 << "n_negative_sample, density_samples, density_positive,"
			 << "density_negative, true_positive, false_positive, true_negative,"
			 << "false_negative, accuracy, n_intersection_with_gt, n_intersection_positive_with_gt,"
			 << "n_intersection_negative_with_gt, n_gt_points, density_intesection_with_gt" << endl;

	file << output_path << ","
		 << image_points << ","
		 << n_samples << ","
		 << n_positive_sample << ","
		 << n_negative_sample << ","
		 << fixed << setw(11) << setprecision(3) << density_samples << ","
		 << fixed << setw(11) << setprecision(3) << density_positive << ","
		 << fixed << setw(11) << setprecision(3) << density_negative << ","
		 << true_positive << ","
		 << false_positive << ","
		 << true_negative << ","
		 << false_negative << ","
		 << fixed << setw(11) << setprecision(2) << accuracy << ","
		 << n_intersection_with_gt << ","
		 << n_intersection_positive_with_gt << ","
		 << n_intersection_negative_with_gt << ","
		 << n_gt_points << ","
		 << fixed << setw(11) << setprecision(2) << n_intersection_with_gt/ (float)n_gt_points << endl;
}

