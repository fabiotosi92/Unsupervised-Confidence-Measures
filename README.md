# Unsupervised-Confidence-Measures

**[Learning confidence measures in the wild](http://vision.deis.unibo.it/~smatt/Papers/BMVC2017/BMVC_2017.pdf)**  
[Fabio Tosi](https://vision.disi.unibo.it/~ftosi/), [Matteo Poggi](https://vision.disi.unibo.it/~mpoggi/), Alessio Tonioni, Luigi Di Stefano and [Stefano Mattoccia](https://vision.disi.unibo.it/~smatt/Site/Home.html)   
BMVC (British Machine Vision Conference) 2017

This strategy provides labels for training confidence measures based on machine-learning technique without ground-truth labels. Compared to state-of-the-art, this method is neither constrained to image sequences nor to image content. 

**Warning:** This C++ implementation contains AD-CENSUS stereo matching algorithm only.

### [KITTI 2012 sample](http://www.cvlibs.net/datasets/kitti/raw_data.php)

![Alt text](https://github.com/fabiotosi92/Unsupervised-Confidence-Measures/blob/master/images/000059_10/disparity.png?raw=true "disparity")
![Alt text](https://github.com/fabiotosi92/Unsupervised-Confidence-Measures/blob/master/images/000059_10/rgb_samples.png?raw=true "rgb_samples")
![Alt text](https://github.com/fabiotosi92/Unsupervised-Confidence-Measures/blob/master/images/000059_10/disparity_positive.png?raw=true "correct_disparities" )

## Requirements
This C++ code is developed under Ubuntu 16.04. The following libraries are required:
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
    -t <threshold> : percentage of pixels considered most/least confident (default=0.4)
    -b <bad> : value used to discriminate positive from negative samples (default=3)
    -g <groundtruth> : if groundtruth is available you can test the accuracy of positive and negative samples
    -f <evaluation_file> : CSV file in which the evaluation will be saved (if -g arg is enabled)
    -i <invalid_pixel> : invalid pixel value for gt image (e.g 0 for KITTI, 255 for Middlebury) 
    -s <scale_factor> : scale factor for gt image (e.g 256 for KITTI, 1 for Middlebury)
Example without groundtruth for test:

```shell
./build/bmvc2017 -l [left_image] -r [right_image] -o [output_path] -p lrc uc dbl apkr med wmn -n lrc uc apkr wmn -t 0.3 -b 3 -d 228
```
Example with groundtruth for test:

```shell
./build/bmvc2017 -l [left_image] -r [right_image] -g [gt_image] -f [csv_file] -o [output_path] -p lrc uc dbl apkr med wmn -n lrc uc apkr wmn -t 0.3 -b 3 -d 228 -i 0 -s 256
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
     - da (Disparity Agreement)
     - ds (Disparity Scattering)
     
For more details about confidence measures: [Quantitative evaluation of confidence measures in a machine learning world](http://vision.deis.unibo.it/~smatt/Papers/ICCV2017/ICCV_2017_confidence_measures.pdf) 

## Reference

If you use this code, please cite our paper: 
```
@inproceedings{BMVC_2017,
   author               = {Tosi, Fabio and Poggi, Matteo and Tonioni, Alessio and Di Stefano, Luigi and Mattoccia, Stefano},
   booktitle            = {28th British Machine Vision Conference (BMVC 2017)},
   month                = {September},
   title                = {Learning confidence measures in the wild},
   year                 = {2017},
   }
```
