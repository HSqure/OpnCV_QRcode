#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MarkerDetector.h"
#include "GeometryTypes.h"
#include "Marker.h"
#include "TinyLA.hpp"

using namespace cv;
using namespace std;

int main()
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}

	bool stop = false;

	while (!stop)
	{
		MarkerDetector markGet;
		Mat frame;//原图
		Mat GRAYframe;//灰度
		Mat PThframe;//二值
		Mat contourImg;//轮廓
		std::vector<std::vector<cv::Point>> m_contours;
		Mat src, dst, bin;
		//vector<vector<Point>> line;
		vector<Marker> markers;
		vector<Point2f> m_markerCorners2d; //标准maker坐标4个点
		Size markerSize(100, 100);			   //标准maker大小
		vector<Marker> detectedMarkers;

		//Mat PROFILEframe;

		frame = imread("C:\\Users\\Mr.HH\\Pictures\\Camera Roll\\maker.png");

		//if (frame.empty())
		//{
		//	cout << "抱歉啦！臣妾没有找到这个文件！" << endl;

		//	return 0;
		//}

		cap >> frame;
		//cout << "图像image大小为" << frame.rows << "x"
		//	<< frame.cols << endl;


		src = frame;
		Mat contours(src.size().height, src.size().width, CV_8UC3, Scalar(0, 0, 0));
		m_markerCorners2d.push_back(Point2f(0, 0));
		m_markerCorners2d.push_back(Point2f(markerSize.width - 1, 0));
		m_markerCorners2d.push_back(Point2f(markerSize.width - 1, markerSize.height - 1));
		m_markerCorners2d.push_back(Point2f(0, markerSize.height - 1));




		//【1】灰度化
		markGet.prepareImage(src, dst);

		//【2】灰度图二值化(自适应阈值)
		markGet.performThreshold(dst, bin);


		//【3】检测二值图边缘
		markGet.findContours(bin, m_contours, GRAYframe.cols / 100);
		vector<Vec4i> hierarchy;
		contourImg = Mat::zeros(frame.size(), CV_8UC3);//清空
		for (int i = 0; i < m_contours.size(); i++)
		{
			drawContours(contourImg, m_contours, i, Scalar(255, 255, 255), 2, 8, hierarchy, 0, Point());
		}
		//【4】
		markGet.findMarkerCandidates(m_contours, detectedMarkers);//line = m_contours

																  // Find is them are markers
		markGet.detectMarkers(dst, detectedMarkers, m_markerCorners2d);

		imshow("图像", dst);
		//imwrite("C:\\Users\\Mr.HH\\Desktop\\maker2.jpg", contourImg);


		//imshow("图像2", contourImg);
		//imwrite("C:\\Users\\Mr.HH\\Desktop\\maze.jpg", contourImg);
		//imshow("当前视频", frame);
		if (waitKey(30) >= 0)
			stop = true;
	}
	waitKey(0);

	return 0;

}
