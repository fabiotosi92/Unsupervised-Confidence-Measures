# Unsupervised-Confidence-Measures
This strategy provides labels for training confidence measures based on machine-learning technique without ground-truth labels. Compared to state-of-the-art, this method is neither constrained to image sequences nor to image content.

**[Learning confidence measures in the wild](http://vision.deis.unibo.it/~smatt/Papers/BMVC2017/BMVC_2017.pdf)**  
Fabio Tosi, Matteo Poggi(https://vision.disi.unibo.it/~mpoggi/), Alessio Tonioni, Luigi Di Stefano and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html)   
BMVC 2017

## Requirements
This C++ code is developed under Ubunutu 16.04. The following libraries are required:
- gcc
- cmake 
- OpenCV (2.4.x)

## Building from the command line
Create a build directory and run cmake there:

 - mkdir build
 - cd build 
 - cmake ..
 - make

## Usage
Input (required):

    -l <left_image> : left image of stereo pair 
    -r <right_image> : right image of stereo pair
    -o <output_path> : output folder  
    -p <positive_cms> : list of confidence measures used to select positive samples
    -n <negative_cms> : list of confidence measures used to select negative samples
    
Input (optional):

    -d <d_max> : max disparity range (default=228)
    -t0 <threshold_0> : percentage of pixels considerated least confident (default=0.4)
    -t1 <threshold_1> : percentage of pixels considerated most confident (default=0.4)
    -b <bad> : value used to discriminate positive from negative samples (default=3)
    -g <groundtruth> : if groundtruth is available you can test the accuracy of positive and negative samples
    -f <evaluation_file> : CSV file in which the evaluation will be saved (if -g arg is enabled)
    -i <invalid_pixel> : invalid pixel value for gt image (e.g 0 for KITTI, 255 for Middlebury) 
    
Example without groundtruth for test:

```shell
./build/bmvc2017 -l [left_image] -r [right_image] -o [outputpath] -p lrc uc dbl apkr med wmn -n lrc uc apkr wmn -t 0.3 -b 3 -d 228
```
Example with groundtruth for test:

```shell
./build/bmvc2017 -l [left_image] -r [right_image] -g [gt_image] -f [csv_file] -o [outputpath] -p lrc uc dbl apkr med wmn -n lrc uc apkr wmn -t 0.3 -b 3 -d 228 -i 0 
```
    
 ## List of usable confidence measures (CMs) 
 
     - msm (Matching Score Measure)
     - lc (Local Curve)
     - cur (Curvature)
     - pkr (Peak Ratio)
     - pkrn (Peak Ratio Naive)
     - apkr (Average Peak Ratio)
     - apkrn (Average Peak Ratio Naive)
     - mm (Maximum Margin)
     - mmn (Maximum Margin Naive)
     - nmm (Nonlinear Maximum Margin)
     - nmmn (Nonlinear Maximum Margin Naive)
     - dvm (Disparity Variance Measure)
     - dam (Disparity Ambiguity Measure)
     - mlm (Maximum Likelihood Measure)
     - aml (Attainable Maximum Likelihood)
     - nem (Negative Entropy Measure)
     - noi (Number Of Inflections)
     - wmn (Winner Margin)
     - wmnn (Winner Margin Naive)
     - per (Perturbation Measure)
     - lrc (Left Right Consistency Check)
     - acc (Asymmetric Consistency Check)
     - uc (Uniqueness Constraint)
     - lrc (Left Right Difference)
     - hgm (Horizontal Gradient Measure)
     - zsad (Zero Mean Sum Of Absolute Difference)
     - dsm (Distinctive Similarity Measure)
     - samm (Self Aware Matching Measure)
     - dbl (Distance To Border Left)
     - dtd (Distance To Discontinuity)
     - dte (Distance To Edge)
     - med (Difference With Median)
     - lmn (Local Minima In Neighborhood)
     
For more details on confidence measures: [Quantitative evaluation of confidence measures in a machine learning world](http://vision.deis.unibo.it/~smatt/Papers/ICCV2017/ICCV_2017_confidence_measures.pdf) 