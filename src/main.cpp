#include "stereo_matching.hpp"
#include "confidence_measures.hpp"
#include "DSI.hpp"
#include "generate_samples.hpp"
#include "evaluation.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <argparse.hpp>
#include <iterator>

bool exists(string filePath)
{
	struct stat st;
	
	if(stat(filePath.c_str(),&st) == 0)
		if(st.st_mode & S_IFDIR != 0)
			return true;
	else false;
}


int main(int argc, const char** argv)
{
	cout << string( 3, '\n' ) << " ***** BMVC 2017 - Learning confidence measures in the wild *****" << endl;
	cout << "   (F. Tosi, M. Poggi, A. Tonioni, L. Di Stefano, S. Mattoccia)" << string( 3, '\n' );

	/***************************************************************************************************/
	/*
	* Parse args
	*
	/***************************************************************************************************/

	ArgumentParser parser;

    parser.addArgument("-l", "--image_left", 1, true);
    parser.addArgument("-r", "--image_right", 1, true);
    parser.addArgument("-o", "--output", 1, true);
    parser.addArgument("-f", "--file_output", 1, true);
    parser.addArgument("-g", "--gt", 1, true);
    parser.addArgument("-t0", "--threshold0", 1);
    parser.addArgument("-t1", "--threshold1", 1);
    parser.addArgument("-d", "--dmax", 1);
    parser.addArgument("-b", "--bad", 1);
    parser.addArgument("-s", "--scale_factor", 1);
    parser.addArgument("-i", "--invalid", 1);
    parser.addArgument("-p", "--confidences_positive", '+');
    parser.addArgument("-n", "--confidences_negative", '+');
	parser.parse( argc, argv );

	string left_path = parser.retrieve<string>("image_left");
	string right_path = parser.retrieve<string>("image_right");
	string output_path = parser.retrieve<string>("output");
	string file_output_path = parser.retrieve<string>("file_output");
	string gt_path = parser.retrieve<string>("gt");
	string t0 = parser.retrieve<string>("threshold0");
	string t1 = parser.retrieve<string>("threshold1");
    string d = parser.retrieve<string>("dmax");
    string b = parser.retrieve<string>("bad");
    string s = parser.retrieve<string>("scale_factor");
    string i = parser.retrieve<string>("invalid");
	vector<string> choices_positive = parser.retrieve< vector<string> >("confidences_positive");
	vector<string> choices_negative = parser.retrieve< vector<string> >("confidences_negative");
	vector<string> choices;

	copy(choices_positive.begin(), choices_positive.end(), back_inserter(choices));
	copy(choices_negative.begin(), choices_negative.end(), back_inserter(choices));
	sort(choices.begin(), choices.end());
    choices.erase(unique(choices.begin(), choices.end()), choices.end());


	/***************************************************************************************************/
	/*
	* Compute disparity map and confidence estimations (for now, only AD-CENSUS algorithm)
	*
	/***************************************************************************************************/

	//read images
	Mat image_0 = imread(left_path, CV_LOAD_IMAGE_UNCHANGED);
	Mat image_1 = imread(right_path, CV_LOAD_IMAGE_UNCHANGED);

	//grayscale
	Mat gray_0, gray_1;
	if(image_0.type() == CV_8UC3 && image_1.type() == CV_8UC3)
	{
		cvtColor(image_0, gray_0, CV_BGR2GRAY);
		cvtColor(image_1, gray_1, CV_BGR2GRAY);
	}
	else
	{
		gray_0 = image_0;
		gray_1 = image_1;
	}


	//stereo matching params 
	int census = 2, boxfilter = 2;
	int d_min = 0, d_max = (d.empty()) ? 228 : strtof((d).c_str(),0);
	int bad = (d.empty()) ? 3 : strtof((b).c_str(),0);
	int scale = (s.empty()) ? 1 : strtof((s).c_str(),0);
	int invalid = (i.empty()) ? 0 : strtof((i).c_str(),0);

	//vector of confidence maps
	Mat disparity_L2R;
	vector<Mat> confidences;
	vector<string> confidence_names; 

	//stereo
    cout << " - generate dsi_LR..." << endl;
	_DSI dsi_LR = SHD_box_filtering(gray_0, gray_1, d_min, d_max, census, boxfilter);

    cout << " - generate dsi_LL..." << endl;
	_DSI dsi_LL = SHD_box_filtering(gray_0, gray_0, (d_min - d_max) / 2, (d_max - d_min) / 2, census, boxfilter);

    cout << " - generate dsi_RR..." << endl;
	_DSI dsi_RR = SHD_box_filtering(gray_1, gray_1, (d_min - d_max) / 2, (d_max - d_min) / 2, census, boxfilter);


	//compute confidence measures
	fn_confidence_measure(gray_0, gray_1, dsi_LR, dsi_LL, dsi_RR, bad, choices, disparity_L2R, confidence_names, confidences);

	/***************************************************************************************************/
	/*
	* Generate training samples
	*
	/***************************************************************************************************/	
	cout << string( 2, '\n' ) 
	     << "***** Generate training samples *****" 
	     << string( 2, '\n' );

	//cast to numbers
	float threshold0 = (t0.empty()) ? 0.4 : strtof((t0).c_str(),0);
	float threshold1 = (t1.empty()) ? 0.4 : strtof((t1).c_str(),0);

	Mat positive_samples, negative_samples;
    generate_training_samples(confidences, disparity_L2R, threshold0, threshold1, confidence_names, choices_positive, choices_negative,
    	positive_samples, negative_samples);

	/***************************************************************************************************/
	/*
	* Save the results
	*
	/***************************************************************************************************/
	cout << string( 2, '\n' ) 
	     << "***** Save the results *****" 
	     << string( 2, '\n' );	

	if(!exists(output_path))
	{
		string command = "mkdir -p " + output_path;
		system(command.c_str());
	}

	Mat rgb;
	samples_on_image(image_0, positive_samples, negative_samples, rgb);
	imwrite(output_path + "/positive_samples.png", positive_samples);
	imwrite(output_path + "/negative_samples.png", negative_samples);
	imwrite(output_path + "/rgb_samples.png", rgb);
	write_disparity_map(disparity_L2R, positive_samples, output_path + "/disparity_positive.png");
	write_disparity_map(disparity_L2R, negative_samples, output_path + "/disparity_negative.png");
	write_disparity_map(disparity_L2R, Mat(), output_path + "/disparity.png");

	/***************************************************************************************************/
	/*
	* Evaluation
	*
	/***************************************************************************************************/
	if(!gt_path.empty())
	{
	    cout << string( 2, '\n' ) 
	         << "***** Evaluation *****" 
	         << string( 2, '\n' );

		eval_and_print(disparity_L2R, positive_samples, negative_samples, 
			           gt_path, bad, invalid, scale, output_path, file_output_path);
	}

    cout << string( 2, '\n' ) 
         << "***** Done! *****" 
         << string( 2, '\n' );

	return 0;
}

