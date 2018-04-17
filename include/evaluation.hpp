#ifndef EVALUATION_H
#define EVALUATION_H

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

void samples_on_image
(
	cv::InputArray left, 
	cv::InputArray positive, 
	cv::InputArray negative,
	cv::OutputArray output
);

void eval_and_print
(
	cv::InputArray disparity, 
	cv::InputArray positive_samples, 
	cv::InputArray negative_samples, 
	std::string gt_path, 
	float bad, 
	float invalid, 
	int scale, 
	std::string output_path,
	std::string file_output_path
);

#endif