#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MarkerDetector.h"
#include "GeometryTypes.h"
#include "Marker.h"
#include "TinyLA.hpp"
#define FOCAL_LENGTH 1000

using namespace cv;
using namespace std;






//【01】灰度图转换函数
void MarkerDetector::prepareImage(const Mat& bgraMat, Mat& grayscale)
{
	cvtColor(bgraMat, grayscale,CV_BGR2GRAY);
}




//【02】灰度图二值化函数(自适应阈值)
void MarkerDetector::performThreshold(const Mat& grayscale, Mat& thresholdImg)
{
	//自适应阈值，利用所有像素在给定的半径周围的检查像素

	//输入图像  
	//输出图像  
	//使用 CV_THRESH_BINARY 和 CV_THRESH_BINARY_INV 的最大值  
	//自适应阈值算法使用：CV_ADAPTIVE_THRESH_MEAN_C 或 CV_ADAPTIVE_THRESH_GAUSSIAN_C   
	//取阈值类型：必须是下者之一  
	//CV_THRESH_BINARY,  
	//CV_THRESH_BINARY_INV  
	//用来计算阈值的像素邻域大小: 3, 5, 7, ...  

	adaptiveThreshold(grayscale, thresholdImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);

}



//【03】轮廓检测函数
void MarkerDetector::findContours(const cv::Mat &thresholdImg, 
									std::vector<std::vector<cv::Point>> &contours, 
										int minContoursPointAllowed)
{
	//这个函数的输出是一个一个多边形的集合，每个多边形多代表一个可能的轮廓。
	//在这个方法中，我们忽略了尺寸小于minContoursPointAllowed的多边形，
	//因为我们认为它们要么不是有效的轮廓，要么实在太小，不值得去检测它们


	//所有的轮廓 
	//vector是可变数组，stl的一种，也就是大小可变的数组
	//vector<Point>的意思是，这个类型是包含了很多"point类型"的集合
	//vector< vector<Point> >的意思是，包含很多vector<Point>的集合，也就是说包含了多个“包含point的集合”的集合
	std::vector<std::vector<cv::Point>> allContours;

	//输入图像image必须为一个二值单通道图像  
	//检测的轮廓数组，每一个轮廓用一个point类型的vector表示  



/*
	轮廓的检索模式：
	
	[1] CV_RETR_EXTERNAL 表示只检测外轮廓

	[2] CV_RETR_LIST 检测的轮廓不建立等级关系

	[3] CV_RETR_CCOMP 建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。
					  如果内孔内还有一个连通物体，这个物体的边界也在顶层。

	[4] CV_RETR_TREE 建立一个等级树结构的轮廓。具体参考contours.c这个demo
*/

/*
	轮廓的近似办法：  
	
	[1] CV_CHAIN_APPROX_NONE 存储所有的轮廓点，相邻的两个点的像素位置差不超过1，
							 即max（abs（x1-x2），abs（y2-y1））==1

	[2] CV_CHAIN_APPROX_SIMPLE 压缩水平方向，垂直方向，对角线方向的元素，
							   只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息

	[3] CV_CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS 使用teh-Chinl chain 近似算法

	[4] offset 表示代表轮廓点的偏移量，可以设置为任意值。对ROI图像中找出的轮廓，并要在整个图像中进行分析时，
			   这个参数还是很有用的。
*/


	cv::findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contours.clear();

	// 做一个筛选,如果一个contour的点的个数比较少,不是一个好contour
	for (size_t i = 0; i < allContours.size(); i++)
	{
		int size = allContours[i].size();
		if (size > minContoursPointAllowed)
		{
			contours.push_back(allContours[i]);
		}
	}
}


//【04】
// Find closed contours that can be approximated with 4 points  
void MarkerDetector::findMarkerCandidates(const vector<vector<Point>>& contours, vector<Marker>& detectedMarkers)
{
	vector<Point>  approxCurve;//相似形状     
	vector<Marker> possibleMarkers;//可能的标记
	float m_minContourLengthAllowed = 100.0;

	//分析每个标记是否是一个类似标记的平行六面体
	for (size_t i = 0; i<contours.size(); i++)
	{
		//通过点集近似多边形，第三个参数为epsilon代表近似程度，
		//即原始轮廓及近似多边形之间的距离，第四个参数表示多边形是闭合的 

		//近似一个多边形  
		double eps = contours[i].size() * 0.05;


		//函数approxPolyDP返回结果为多边形，用点集表示 
		//使多边形边缘平滑，得到近似的多边形
		approxPolyDP(contours[i], approxCurve, eps, true);

		//这里只考虑四边形（即只包含四个顶点的多边形）
		if (approxCurve.size() != 4)
			continue;//如果满足条件，该语句为本次循环的终点，调回for开头

		// 而且必须是凸面的 
		if (!isContourConvex(approxCurve))
			continue;

		//确保相邻的两点间的距离“足够大”－大到是一条边而不是短线段就是了
		//float minDist = 1e10;
		float minDist = std::numeric_limits<float>::max();
		for (int i = 0;i<4;i++)
		{
			//求当前四边形各顶点之间最短距离  
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredDistance = side.dot(side);
			//取最小值
			minDist = min(minDist, squaredDistance);
		}

		// 检查之间距离是否过小，当四边形大小合适时，则将该四边形maker放入possibleMarkers容器内  
		if (minDist > m_minContourLengthAllowed)
		{
			//保存相似的标记
			Marker m;
			for (int i = 0;i<4;i++)
			{
				m.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));
			}
			possibleMarkers.push_back(m);
		}

	}




	//逆时针顺序存储这些坐标点  
	for (size_t i = 0; i<possibleMarkers.size(); i++)
	{
		Marker& marker = possibleMarkers[i];

		//trace a line between the first and second point.  
		//if the thrid point is at the right side, then the points are anti-clockwise  
		//从代码推测，marker中的点集本来就两种序列：顺时针和逆时针，这里要把顺时针的序列改成逆时针  
		Point v1 = marker.points[1] - marker.points[0];
		Point v2 = marker.points[2] - marker.points[0];

		//行列式的几何意义是什么呢？有两个解释：一个解释是行列式就是行列式中的行或列向量所构成的
		//超平行多面体的有向面积或有向体积；另一个解释是矩阵A的行列式detA就是线性变换A下的图形面积
		//或体积的伸缩因子。  
		double o = (v1.x * v2.y) - (v1.y * v2.x);
/*

		以行向量a=(a1,a2)，b=(b1,b2)为邻边的平行四边形的有向面积：若这个平行四边形是由向量沿逆时针方向
		转到b而得到的，面积取正值；若这个平行四边形是由向量a沿顺时针方向转到而得到的，面积取负值；
		以下代码即把这种改成逆时针

		0----1
		 \
		   \
		3    2


*/


		//如果第三点在左侧，则按逆时针顺序排序  
		if (o < 0.0)
		{
			//关于swap函数的资料：http://blog.csdn.net/dizuo/article/details/6435847
			//交换两点的位置
			swap(marker.points[1], marker.points[3]);
		}
	}

	//移除角落太接近的元素  
	//第一次检测相似性 
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0;i<possibleMarkers.size();i++)
	{
		const Marker& m1 = possibleMarkers[i];

		//计算每一个角落的平均距离到最近的一个角落的其他标记候选  
		//计算两个maker四边形之间的距离，四组点之间距离和的平均值，若平均值较小，则认为两个maker很相近  

		for (size_t j = i + 1;j<possibleMarkers.size();j++)
		{
			const Marker& m2 = possibleMarkers[j];

			float distSquared = 0;

			for (int c = 0;c<4;c++)
			{
				Point v = m1.points[c] - m2.points[c];
				//向量的点乘 ---> 两点的距离
				distSquared += v.dot(v);
			}

			distSquared /= 4;

			if (distSquared < 100)
			{
				tooNearCandidates.push_back(std::pair<int, int>(i, j));
			}
		}
	}

	//mark for removal the element of  the pair with smaller perimeter  
	//计算距离相近的两个marker内部，四个点的距离和，将距离和较小的，在removlaMask内做标记，
	//即不作为最终的detectedMarkers  
	std::vector<bool> removalMask(possibleMarkers.size(), false);

	for (size_t i = 0;i<tooNearCandidates.size();i++)
	{	//周长
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		removalMask[removalIndex] = true;
	}

	//返回可能的对象  
	detectedMarkers.clear();
	for (size_t i = 0;i<possibleMarkers.size();i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);/*detectedMarkers为出参*/
	}
}



//【05】
void MarkerDetector::detectMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers, const vector<Point2f> m_markerCorners2d)
{
	Mat canonicalMarker;
	char name[50] = "";

	vector<Marker> goodMarkers;

	//识别标记
	for (size_t i = 0;i<detectedMarkers.size();i++)
	{
		Marker& marker = detectedMarkers[i];
		// Find the perspective transfomation that brings current marker to rectangular form
		// 得到当前marker的透视变换矩阵M
		//它首先根据四个对应的点找到透视变换，第一个参数是标记的坐标，第二个是正方形标记图像的坐标。
		//估算的变换将会把标记转换成方形，从而方便我们分析。
		// 找到透视投影，并把标记转换成矩形  
		//输入图像四边形顶点坐标  
		//输出图像的相应的四边形顶点坐标 
		Mat M = getPerspectiveTransform(marker.points, m_markerCorners2d);


		// Transform image to get a canonical marker image
		// 将当前的marker变换为正交投影
		//输入的图像  
		//输出的图像  
		//3x3变换矩阵 
		warpPerspective(grayscale, canonicalMarker, M, Size(100, 100));
		
		//imshow("视频2", canonicalMarker);//试看效果,小方块原图

		//这是固定阀值方法  
		//输入图像image必须为一个2值单通道图像  
		//检测的轮廓数组，每一个轮廓用一个point类型的vector表示   
		//阀值max_value 使用 CV_THRESH_BINARY 和 CV_THRESH_BINARY_INV 的最大值   
		cv::threshold(canonicalMarker, canonicalMarker, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		//sprintf(name,"C:\\Users\\Mr.HH\\Desktop\\maker\\warp_%d.jpg",i);
		imshow("二维码", canonicalMarker);

		int nRotations;
		int id = Marker::getMarkerId(canonicalMarker, nRotations);//5*5 mask,3 bits校验,2 bits 数据,每个stripe有4种数据,共有5条stripe,故有4^5个ID,白块为1，黑块为0
		cout<<"ID:"<<id<<endl;
		if (id != -1)
		{
			marker.id = id;
			//sort the points so that they are always in the same order no matter the camera orientation
			//Rotates the order of the elements in the range [first,last), in such a way that the element pointed by middle becomes the new first element.
			rotate(marker.points.begin(), marker.points.begin() + 4 - nRotations, marker.points.end());

			goodMarkers.push_back(marker);
		}

	}

	//refine using subpixel accuracy the corners
	if (goodMarkers.size() > 0)
	{
		std::vector<Point2f> preciseCorners(4 * goodMarkers.size());//每个marker四个点

		for (size_t i = 0; i<goodMarkers.size(); i++)
		{
			Marker& marker = goodMarkers[i];

			for (int c = 0;c<4;c++)
			{
				preciseCorners[i * 4 + c] = marker.points[c];	//i表示第几个marker，c表示某个marker的第几个点
			}
		}

		//Refines the corner locations.The function iterates to find the sub-pixel accurate location of corners or radial saddle points
		cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
		cv::cornerSubPix(grayscale, preciseCorners, cvSize(5, 5), cvSize(-1, -1), termCriteria);

		//copy back
		for (size_t i = 0;i<goodMarkers.size();i++)
		{
			Marker& marker = goodMarkers[i];

			for (int c = 0;c<4;c++)
			{
				marker.points[c] = preciseCorners[i * 4 + c];
				cout<<"X:"<<marker.points[c].x<<"Y:"<<marker.points[c].y<<endl;
			}
		}
	}

	//画出细化后的矩形图片
	//新建一个图像容器并标定大小，大小等于输入图像
	cv::Mat markerCornersMat(grayscale.size(), grayscale.type());
	markerCornersMat = cv::Scalar(0);//初始化为0

	for (size_t i=0; i<goodMarkers.size(); i++)
	{
	goodMarkers[i].drawContour(markerCornersMat, cv::Scalar(255));
	}

	imshow("refine",markerCornersMat);

	detectedMarkers = goodMarkers;

}



//【06】
void MarkerDetector::estimatePosition(vector<Marker>& detectedMarkers, const vector<Point3f> &m_markerCorners3d, const Matrix33 m_intrinsic, const Vector4  m_distorsion)
{
	Mat camMatrix;
	Mat distCoeff;

	Mat(3, 3, CV_32F, const_cast<float*>(&m_intrinsic.data[0])).copyTo(camMatrix);
	Mat(4, 1, CV_32F, const_cast<float*>(&m_distorsion.data[0])).copyTo(distCoeff);

	for (size_t i = 0; i<detectedMarkers.size(); i++)
	{
		Marker& m = detectedMarkers[i];

		Mat Rvec;
		Mat_<float> Tvec;
		Mat raux, taux;
		solvePnP(m_markerCorners3d, m.points, camMatrix, distCoeff, raux, taux);
		raux.convertTo(Rvec, CV_32F);
		taux.convertTo(Tvec, CV_32F);

		Mat_<float> rotMat(3, 3);
		Rodrigues(Rvec, rotMat);
		// Copy to transformation matrix
		m.transformation = Transformation();

		for (int col = 0; col<3; col++)
		{
			for (int row = 0; row<3; row++)
			{
				m.transformation.r().mat[row][col] = rotMat(row, col); // Copy rotation component
			}
			m.transformation.t().data[col] = Tvec(col); // Copy translation component
		}

		// Since solvePnP finds camera location, w.r.t to marker pose, to get marker pose w.r.t to the camera we invert it.
		m.transformation = m.transformation.getInverted();
	}
}