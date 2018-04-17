#ifndef GENERATE_SAMPLES_H
#define GENERATE_SAMPLES_H

#include "confidence_measures_utility.hpp"
#include "sort.hpp"

#include <opencv2/opencv.hpp>

void generate_training_samples
(
	cv::InputArrayOfArrays confidence_measures, 
	cv::InputArray disparity_map, 
	float threshold0, float threshold1,
	std::vector<std::string> confidence_names, 
	std::vector<std::string> choices_positive, 
	std::vector<std::string> choices_negative, 
	cv::OutputArray correct, 
	cv::OutputArray wrong
);

#endif
