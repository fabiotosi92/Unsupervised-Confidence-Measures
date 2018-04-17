#ifndef _CONFIDENCE_MEASURES
#define _CONFIDENCE_MEASURES

#include "confidence_measures_utility.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct confParams
{
	int zsad_radius,
		apkr_radius,
		lmn_radius,
		med_radius,
		pdwc_radius,
		DD_edge_threshold,
		DD_ratio,
		DD_radius,
		DD_max_lowThreshold,
		dvm;

	float pkr_epsilon,
		  apkr_epsilon,
		  dsm_epsilon,
		  lrd_epsilon,
		  mlm_sigma,
		  aml_sigma,
		  per_sigma,
		  nmm_sigma,
		  lc_gamma;

	confParams() :  
		zsad_radius(-1),
		apkr_radius(-1),
		lmn_radius(-1),
		med_radius(-1),
		pdwc_radius(-1),
		dvm(-1),
		pkr_epsilon(-1),
		apkr_epsilon(-1),
		dsm_epsilon(-1),
		lrd_epsilon(-1),
		mlm_sigma(-1),
		aml_sigma(-1),
		per_sigma(-1),
		nmm_sigma(-1),
		lc_gamma(-1),
		DD_edge_threshold(-1),
		DD_ratio(-1),
		DD_radius(-1) {}
};

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
);

//2.2	Local properties of the cost curve
void curvature
(
	InputArrayOfArrays costs, 
	InputArray c1_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void local_curve
(
	InputArrayOfArrays costs, 
	InputArray c1_idx, 
	float gamma, 
	int height, 
	int width, 
	OutputArray confidence_map
);

//2.3	Local minima of the cost curve
void peak_ratio
(
	InputArray c1, 
    InputArray c2m, 
    float epsilon, 
    int height, 
    int width, 
    OutputArray confidence_map
);

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
);

void maximum_margin
(
	InputArray c1, 
	InputArray c2m, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void non_linear_margin
(
	InputArray c1, 
	InputArray c2m, 
	float sigma, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void disparity_variance_measure
(
	InputArray disparity_map, 
	int size, 
	OutputArray confidence_map
);

void var_measure
(
	InputArray disparity_map, 
	int r, 
	OutputArray confidence_map
);

void disparity_ambiguity_measure
(
	InputArray c1_idx, 
	InputArray c2_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
);

//2.4	The Entire Cost Curve
void winner_margin
(
	InputArray c1, 
	InputArray c2m, 
	InputArray c_sum, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void maximum_likelihood_measure
(
	InputArrayOfArrays costs, 
	InputArray c1, 
	float sigma, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void attainable_maximum_likelihood
(
	InputArrayOfArrays costs, 
	InputArray c1, 
	float sigma, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void negative_entropy_measure
(
	InputArrayOfArrays costs, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void perturbation_measure
(
	InputArrayOfArrays costs, 
	InputArray c1, 
	InputArray c1_idx, 
	float s, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void local_minima_in_neighborhood
(
	InputArrayOfArrays local_minima, 
	InputArray c1_idx, 
	int r, 
	int height, 
	int width, 
	OutputArray confidence_map
);

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
);

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
);

void uniqueness_constraint
(
	InputArray c1, 
	InputArray c1_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void asymmetric_consistency_check
(
	InputArray c1, 
	InputArray c1_idx, 
	int height, 
	int width, 
	OutputArray confidence_map
);

//2.6	Matching cost between left and right image intensities
void horizontal_gradient
(
	InputArray left_stereo, 
	OutputArray confidence_map
);

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
);

//2.7	Distinctiveness-based confidence measures
void distinctiveness
(
	InputArrayOfArrays costs_LL, 
	int d_min, 
	int d_max, 
	int height, 
	int width, 
	OutputArray confidence_map
);

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
);

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
);

//2.8  Based on image information
void distance_to_border
(
	InputArray image,
	OutputArray confidence_map
);

void distance_to_left_border
(
	InputArray image, 
	int d_max, 
	OutputArray confidence_map
);

void compute_DD
(
	InputArray image, 
	int lowThreshold, 
	int ratio, 
	int r, 
	OutputArray confidence_map
);

//2.9 Based on disparity map
void difference_with_median
(
	InputArray disparity_L2R, 
	int r, 
	int bad, 
	int height, 
	int width, 
	OutputArray confidence_map
);

void error_to_groundtruth
(
	InputArray disparity_L2R, 
	InputArray groundtruth, 
	int height, 
	int width, 
	OutputArray confidence_map
);

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
);

#endif 
