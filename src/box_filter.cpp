#include "box_filter.hpp"

using namespace cv;

void cumsum
(
	InputArray imSrc, 
	int n, 
	OutputArray imDst
){
	Mat _imSrc = imSrc.getMat();

	imDst.create(_imSrc.size(), _imSrc.type());
	Mat _imDst = imDst.getMat();
	_imSrc.copyTo(_imDst);

	int height = _imSrc.rows;
	int width = _imSrc.cols;

	if (n == 1)
	{
		for (int row = 1; row< height; row++)
		{
			float *ImDst_ptr0 = (float*)_imDst.ptr<float>(row);
			float *ImDst_ptr1 = (float*)_imDst.ptr<float>(row - 1); 

			for (int col = 0; col<width; col++)
			{
				ImDst_ptr0[col] = ImDst_ptr1[col] + ImDst_ptr0[col]; 
			}
		}
	}
	else if (n == 2)
	{
		for (int row = 0; row<height; row++)
		{
			float *ImDst_ptr = (float*)_imDst.ptr<float>(row);

			for (int col = 1; col<width; col++)
			{
				ImDst_ptr[col] = ImDst_ptr[col - 1] + ImDst_ptr[col];
			}
		}
	}
}

void box_filter
(
	InputArray imSrc, 
	int r, 
	OutputArray imDst
){
	Mat _imSrc = imSrc.getMat();

	int height = _imSrc.rows;
	int width = _imSrc.cols;

	imDst.create(_imSrc.size(), CV_32F);
	Mat _imDst = imDst.getMat();

	/****************************/
	/*CUMULATIVE SUM OVER Y AXIS*/
	/****************************/

	Mat _imCum;
	cumsum(_imSrc, 1, _imCum);

	for (int row = 0; row < r + 1; row++)
	{
		float *imDst_ptr = (float*)_imDst.ptr<float>(row);   
		float *imCum_ptr = (float*)_imCum.ptr<float>(row + r); 

		for (int col = 0; col<width; col++)
		{
			imDst_ptr[col] = imCum_ptr[col]; 
		}
	}

	for (int row = r + 1; row<height - r; row++)
	{
		float *imDst_ptr = (float*)_imDst.ptr<float>(row);
		float *imCum_ptr0 = (float*)_imCum.ptr<float>(row + r); 
		float *imCum_ptr1 = (float*)_imCum.ptr<float>(row - (r + 1));

		for (int col = 0; col<width; col++)
		{
			imDst_ptr[col] = imCum_ptr0[col] - imCum_ptr1[col]; 
		}
	}

	Mat _subImage = Mat(Size(_imSrc.cols, r), CV_32F);
	
	for (int row = 0; row < r; row++)
	{
		float *subImage_ptr = (float*)_subImage.ptr<float>(row);
		float *imCum_ptr = (float*)_imCum.ptr<float>(height - 1);

		for (int col = 0; col<width; col++)
		{
			subImage_ptr[col] = imCum_ptr[col]; 
		}
	}

	for (int row = height - r; row < height; row++)
	{
		float *imDst_ptr = (float*)_imDst.ptr<float>(row);
		float *subImage_ptr = (float*)_subImage.ptr<float>(row + r - height);
		float *imCum_ptr = (float*)_imCum.ptr<float>(row - r - 1);

		for (int col = 0; col < width; col++)
		{
			imDst_ptr[col] = subImage_ptr[col] - imCum_ptr[col]; 
		}
	}

	_imCum.release();
	_subImage.release();


	/****************************/
	/*CUMULATIVE SUM OVER X AXIS*/
	/****************************/

	cumsum(_imDst, 2, _imCum);

	for (int row = 0; row<height; row++)
	{
		float *imDst_ptr = (float*)_imDst.ptr<float>(row);
		float *imCum_ptr = (float*)_imCum.ptr<float>(row);

		for (int col = 0; col < r + 1; col++)
		{
			imDst_ptr[col] = imCum_ptr[(col + r)]; 
		}
	}

	for (int row = 0; row < height; row++)
	{
		float *imDst_ptr = (float*)_imDst.ptr<float>(row);
		float *imCum_ptr = (float*)_imCum.ptr<float>(row);

		for (int col = r + 1; col < width - r; col++)
		{
			imDst_ptr[col] = imCum_ptr[(col + r)] - imCum_ptr[(col - (r + 1))]; 
		}
	}

	_subImage = Mat(Size(r, _imSrc.rows), CV_32F);
	
	for (int row = 0; row < height; row++)
	{
		float *subImage_ptr = (float*)_subImage.ptr<float>(row);
		float *imCum_ptr = (float*)_imCum.ptr<float>(row);

		for (int col = 0; col < r; col++)
		{
			subImage_ptr[col] = imCum_ptr[(width - 1)];
		}
	}

	for (int row = 0; row<height; row++)
	{
		float *imDst_ptr = (float*)_imDst.ptr<float>(row);
		float *subImage_ptr = (float*)_subImage.ptr<float>(row);
		float *imCum_ptr = (float*)_imCum.ptr<float>(row);

		for (int col = width - r; col< width; col++)
		{
			imDst_ptr[col] = subImage_ptr[(col + r - width)] - imCum_ptr[(col - (r + 1))]; 
		}
	}

	_subImage.release();
	_imCum.release();
}