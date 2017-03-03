/*****************************************************************************
*   Marker.cpp
*   Example_MarkerBasedAR
******************************************************************************
*   by Khvedchenia Ievgen, 5th Dec 2012
*   http://computer-vision-talks.com
******************************************************************************
*   Ch2 of the book "Mastering OpenCV with Practical Computer Vision Projects"
*   Copyright Packt Publishing 2012.
*   http://www.packtpub.com/cool-projects-with-opencv/book
*****************************************************************************/

#include "Marker.h"

Marker::Marker() : id(-1)
{
}

bool operator<(const Marker &M1, const Marker&M2)
{
	return M1.id<M2.id;
}

cv::Mat Marker::rotate(cv::Mat in)
{
	cv::Mat out;
	in.copyTo(out);
	for (int i = 0;i<in.rows;i++)
	{
		for (int j = 0;j<in.cols;j++)
		{
			out.at<uchar>(i, j) = in.at<uchar>(in.cols - j - 1, i);
		}
	}
	return out;
}

int Marker::hammDistMarker(cv::Mat bits)
{
	//该矩阵产生：每条stripe有4种可能的数据
	int ids[4][5] =
	{
		{ 1,0,0,0,0 },
		{ 1,0,1,1,1 },
		{ 0,1,0,0,1 },
		{ 0,1,1,1,0 }
	};

	int dist = 0;

	for (int y = 0;y<5;y++)
	{
		int minSum = 1e5; //hamming distance to each possible word

		for (int p = 0;p<4;p++)
		{
			int sum = 0;
			//now, count
			for (int x = 0;x<5;x++)
			{
				sum += bits.at<uchar>(y, x) == ids[p][x] ? 0 : 1;  //拿bitMatrix中每一行同ids中的行依次比较，寻找ids中最贴近bitMatrix第y行的一行ids
			}

			if (minSum>sum)
				minSum = sum;
		}

		//do the and
		dist += minSum;
	}

	return dist;
}

int Marker::mat2id(const cv::Mat &bits)
{
	int val = 0;
	for (int y = 0;y<5;y++)
	{
		val <<= 1;
		if (bits.at<uchar>(y, 1)) val |= 1;
		val <<= 1;
		if (bits.at<uchar>(y, 3)) val |= 1;
	}
	return val;
}

int Marker::getMarkerId(cv::Mat &markerImage, int &nRotations)
{
	//assert(markerImage.rows == markerImage.cols);
	//assert(markerImage.type() == CV_8UC1);

	cv::Mat grey = markerImage;
	//threshold image,参数threshold:阈值125 max_value:255，使用 CV_THRESH_BINARY 和 CV_THRESH_BINARY_INV 的最大值
	cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	//Markers  are divided in 7x7 regions, of which the inner 5x5 belongs to marker info
	//the external border should be entirely black

	int cellSize = markerImage.rows / 7;

	//检查四周边缘
	for (int y = 0;y<7;y++)  //y:row，x:col
	{
		int inc = 6;

		if (y == 0 || y == 6) inc = 1; //for first and last row, check the whole border

									   //第2,3,4,5,6行，只检查左右两列，第1,7行检查所有列
		for (int x = 0;x<7;x += inc)
		{
			int cellX = x * cellSize;
			int cellY = y * cellSize;
			cv::Mat cell = grey(cv::Rect(cellX, cellY, cellSize, cellSize));

			int nZ = cv::countNonZero(cell);

			if (nZ >(cellSize*cellSize) / 2)
			{
				return -1;//can not be a marker because the border element is not black!
			}
		}
	}

	cv::Mat bitMatrix = cv::Mat::zeros(5, 5, CV_8UC1);

	//get information(for each inner square, determine if it is  black or white)  
	for (int y = 0;y<5;y++)
	{
		for (int x = 0;x<5;x++)
		{
			int cellX = (x + 1)*cellSize;
			int cellY = (y + 1)*cellSize;
			cv::Mat cell = grey(cv::Rect(cellX, cellY, cellSize, cellSize));

			int nZ = cv::countNonZero(cell);
			if (nZ>(cellSize*cellSize) / 2)
				bitMatrix.at<uchar>(y, x) = 1;
		}
	}

	//check all possible rotations
	cv::Mat rotations[4];
	int distances[4];

	rotations[0] = bitMatrix;
	distances[0] = hammDistMarker(rotations[0]);

	std::pair<int, int> minDist(distances[0], 0);

	for (int i = 1; i<4; i++)
	{
		//get the hamming distance to the nearest possible word
		rotations[i] = rotate(rotations[i - 1]);
		distances[i] = hammDistMarker(rotations[i]);

		if (distances[i] < minDist.first)
		{
			minDist.first = distances[i];
			minDist.second = i;
		}
	}

	nRotations = minDist.second;
	if (minDist.first == 0)
	{
		return mat2id(rotations[minDist.second]);
	}

	return -1;
}



void Marker::drawContour(cv::Mat& image, cv::Scalar color) const
{
	float thickness = 2;



/*
	imge: 图像
	pt1: 线段的第一个端点。
	pt2: 线段的第二个端点。
	color: 线段的颜色。
	thickness: 线段的粗细程度。
	CV_AA:  antialiased 线条

*/
	cv::line(image, points[0], points[1], color, thickness, CV_AA);
	cv::line(image, points[1], points[2], color, thickness, CV_AA);
	cv::line(image, points[2], points[3], color, thickness, CV_AA);
	cv::line(image, points[3], points[0], color, thickness, CV_AA);
}
