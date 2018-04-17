#include "sort.hpp"

using namespace cv;
using namespace std;

int huge = 0;

void Merge(float confidence[], float rows_mem[], float cols_mem[], int p, int q, int r, int size) 
{
	int i, j, k; 
	if (huge == 0)
	{
		float conf_B[size];
		float rows_B[size];
		float cols_B[size];

		i = p;
		j = q+1;
		k = 0;
		while (i<=q && j<=r)
		{
			if (confidence[i]<confidence[j]) 
			{
				conf_B[k] = confidence[i];
				rows_B[k] = rows_mem[i];
				cols_B[k] = cols_mem[i];
				i++;
			} 
			else 
			{
				conf_B[k] = confidence[j];
				rows_B[k] = rows_mem[j];
				cols_B[k] = cols_mem[j];
				j++;
			}
			k++;
		}
		while (i<=q) 
		{
			conf_B[k] = confidence[i];
			rows_B[k] = rows_mem[i];
			cols_B[k] = cols_mem[i];
			i++;
			k++;
		}
		while (j<=r) 
		{
			conf_B[k] = confidence[j];
			rows_B[k] = rows_mem[j];
			cols_B[k] = cols_mem[j];
			j++;
			k++;
		}
		for (k=p; k<=r; k++)
		{
			confidence[k] = conf_B[k-p];
			rows_mem[k] = rows_B[k-p];
			cols_mem[k] = cols_B[k-p];
		}
	}
	else
	{
		float *conf_B = (float*)malloc(size*sizeof(float));
		float *rows_B = (float*)malloc(size*sizeof(float));		
		float *cols_B = (float*)malloc(size*sizeof(float));	

		i = p;
		j = q+1;
		k = 0;
		while (i<=q && j<=r)
		{
			if (confidence[i]<confidence[j]) 
			{
				conf_B[k] = confidence[i];
				rows_B[k] = rows_mem[i];
				cols_B[k] = cols_mem[i];
				i++;
			} 
			else 
			{
				conf_B[k] = confidence[j];
				rows_B[k] = rows_mem[j];
				cols_B[k] = cols_mem[j];
				j++;
			}
			k++;
		}
		while (i<=q) 
		{
			conf_B[k] = confidence[i];
			rows_B[k] = rows_mem[i];
			cols_B[k] = cols_mem[i];
			i++;
			k++;
		}
		while (j<=r) 
		{
			conf_B[k] = confidence[j];
			rows_B[k] = rows_mem[j];
			cols_B[k] = cols_mem[j];
			j++;
			k++;
		}
		for (k=p; k<=r; k++)
		{
			confidence[k] = conf_B[k-p];
			rows_mem[k] = rows_B[k-p];
			cols_mem[k] = cols_B[k-p];
		}

		free(conf_B);
		free(rows_B);
		free(cols_B);
	}
}

void MergeSort(float confidence[], float rows_mem[], float cols_mem[], int p, int r, int size) 
{
	int q;

	if (p<r) 
	{
		q = (p+r)/2;
		MergeSort(confidence,rows_mem, cols_mem, p, q, size);
		MergeSort(confidence, rows_mem, cols_mem, q+1, r, size);
		Merge(confidence, rows_mem, cols_mem, p, q, r, size);
	}
}

void reverse(float* confidences, float rows_mem[], float cols_mem[], int len)
{
	float t;
	for (int i = 0; i < len / 2; i ++)
	{
		t = rows_mem[i];
		rows_mem[i] = rows_mem[len - 1 - i];
		rows_mem[len - 1 - i] = t;

		t = cols_mem[i];
		cols_mem[i] = cols_mem[len - 1 - i];
		cols_mem[len - 1 - i] = t;

		t = confidences[i];
		confidences[i] = confidences[len - 1 - i];
		confidences[len - 1 - i] = t;
	}
}

