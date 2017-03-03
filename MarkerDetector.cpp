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






//��01���Ҷ�ͼת������
void MarkerDetector::prepareImage(const Mat& bgraMat, Mat& grayscale)
{
	cvtColor(bgraMat, grayscale,CV_BGR2GRAY);
}




//��02���Ҷ�ͼ��ֵ������(����Ӧ��ֵ)
void MarkerDetector::performThreshold(const Mat& grayscale, Mat& thresholdImg)
{
	//����Ӧ��ֵ���������������ڸ����İ뾶��Χ�ļ������

	//����ͼ��  
	//���ͼ��  
	//ʹ�� CV_THRESH_BINARY �� CV_THRESH_BINARY_INV �����ֵ  
	//����Ӧ��ֵ�㷨ʹ�ã�CV_ADAPTIVE_THRESH_MEAN_C �� CV_ADAPTIVE_THRESH_GAUSSIAN_C   
	//ȡ��ֵ���ͣ�����������֮һ  
	//CV_THRESH_BINARY,  
	//CV_THRESH_BINARY_INV  
	//����������ֵ�����������С: 3, 5, 7, ...  

	adaptiveThreshold(grayscale, thresholdImg, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 7);

}



//��03��������⺯��
void MarkerDetector::findContours(const cv::Mat &thresholdImg, 
									std::vector<std::vector<cv::Point>> &contours, 
										int minContoursPointAllowed)
{
	//��������������һ��һ������εļ��ϣ�ÿ������ζ����һ�����ܵ�������
	//����������У����Ǻ����˳ߴ�С��minContoursPointAllowed�Ķ���Σ�
	//��Ϊ������Ϊ����Ҫô������Ч��������Ҫôʵ��̫С����ֵ��ȥ�������


	//���е����� 
	//vector�ǿɱ����飬stl��һ�֣�Ҳ���Ǵ�С�ɱ������
	//vector<Point>����˼�ǣ���������ǰ����˺ܶ�"point����"�ļ���
	//vector< vector<Point> >����˼�ǣ������ܶ�vector<Point>�ļ��ϣ�Ҳ����˵�����˶��������point�ļ��ϡ��ļ���
	std::vector<std::vector<cv::Point>> allContours;

	//����ͼ��image����Ϊһ����ֵ��ͨ��ͼ��  
	//�����������飬ÿһ��������һ��point���͵�vector��ʾ  



/*
	�����ļ���ģʽ��
	
	[1] CV_RETR_EXTERNAL ��ʾֻ���������

	[2] CV_RETR_LIST ���������������ȼ���ϵ

	[3] CV_RETR_CCOMP ���������ȼ��������������һ��Ϊ��߽磬�����һ��Ϊ�ڿ׵ı߽���Ϣ��
					  ����ڿ��ڻ���һ����ͨ���壬�������ı߽�Ҳ�ڶ��㡣

	[4] CV_RETR_TREE ����һ���ȼ����ṹ������������ο�contours.c���demo
*/

/*
	�����Ľ��ư취��  
	
	[1] CV_CHAIN_APPROX_NONE �洢���е������㣬���ڵ������������λ�ò����1��
							 ��max��abs��x1-x2����abs��y2-y1����==1

	[2] CV_CHAIN_APPROX_SIMPLE ѹ��ˮƽ���򣬴�ֱ���򣬶Խ��߷����Ԫ�أ�
							   ֻ�����÷�����յ����꣬����һ����������ֻ��4����������������Ϣ

	[3] CV_CHAIN_APPROX_TC89_L1��CV_CHAIN_APPROX_TC89_KCOS ʹ��teh-Chinl chain �����㷨

	[4] offset ��ʾ�����������ƫ��������������Ϊ����ֵ����ROIͼ�����ҳ�����������Ҫ������ͼ���н��з���ʱ��
			   ����������Ǻ����õġ�
*/


	cv::findContours(thresholdImg, allContours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	contours.clear();

	// ��һ��ɸѡ,���һ��contour�ĵ�ĸ����Ƚ���,����һ����contour
	for (size_t i = 0; i < allContours.size(); i++)
	{
		int size = allContours[i].size();
		if (size > minContoursPointAllowed)
		{
			contours.push_back(allContours[i]);
		}
	}
}


//��04��
// Find closed contours that can be approximated with 4 points  
void MarkerDetector::findMarkerCandidates(const vector<vector<Point>>& contours, vector<Marker>& detectedMarkers)
{
	vector<Point>  approxCurve;//������״     
	vector<Marker> possibleMarkers;//���ܵı��
	float m_minContourLengthAllowed = 100.0;

	//����ÿ������Ƿ���һ�����Ʊ�ǵ�ƽ��������
	for (size_t i = 0; i<contours.size(); i++)
	{
		//ͨ���㼯���ƶ���Σ�����������Ϊepsilon������Ƴ̶ȣ�
		//��ԭʼ���������ƶ����֮��ľ��룬���ĸ�������ʾ������Ǳպϵ� 

		//����һ�������  
		double eps = contours[i].size() * 0.05;


		//����approxPolyDP���ؽ��Ϊ����Σ��õ㼯��ʾ 
		//ʹ����α�Եƽ�����õ����ƵĶ����
		approxPolyDP(contours[i], approxCurve, eps, true);

		//����ֻ�����ı��Σ���ֻ�����ĸ�����Ķ���Σ�
		if (approxCurve.size() != 4)
			continue;//������������������Ϊ����ѭ�����յ㣬����for��ͷ

		// ���ұ�����͹��� 
		if (!isContourConvex(approxCurve))
			continue;

		//ȷ�����ڵ������ľ��롰�㹻�󡱣�����һ���߶����Ƕ��߶ξ�����
		//float minDist = 1e10;
		float minDist = std::numeric_limits<float>::max();
		for (int i = 0;i<4;i++)
		{
			//��ǰ�ı��θ�����֮����̾���  
			Point side = approxCurve[i] - approxCurve[(i + 1) % 4];
			float squaredDistance = side.dot(side);
			//ȡ��Сֵ
			minDist = min(minDist, squaredDistance);
		}

		// ���֮������Ƿ��С�����ı��δ�С����ʱ���򽫸��ı���maker����possibleMarkers������  
		if (minDist > m_minContourLengthAllowed)
		{
			//�������Ƶı��
			Marker m;
			for (int i = 0;i<4;i++)
			{
				m.points.push_back(Point2f(approxCurve[i].x, approxCurve[i].y));
			}
			possibleMarkers.push_back(m);
		}

	}




	//��ʱ��˳��洢��Щ�����  
	for (size_t i = 0; i<possibleMarkers.size(); i++)
	{
		Marker& marker = possibleMarkers[i];

		//trace a line between the first and second point.  
		//if the thrid point is at the right side, then the points are anti-clockwise  
		//�Ӵ����Ʋ⣬marker�еĵ㼯�������������У�˳ʱ�����ʱ�룬����Ҫ��˳ʱ������иĳ���ʱ��  
		Point v1 = marker.points[1] - marker.points[0];
		Point v2 = marker.points[2] - marker.points[0];

		//����ʽ�ļ���������ʲô�أ����������ͣ�һ������������ʽ��������ʽ�е��л������������ɵ�
		//��ƽ�ж��������������������������һ�������Ǿ���A������ʽdetA�������Ա任A�µ�ͼ�����
		//��������������ӡ�  
		double o = (v1.x * v2.y) - (v1.y * v2.x);
/*

		��������a=(a1,a2)��b=(b1,b2)Ϊ�ڱߵ�ƽ���ı��ε���������������ƽ���ı���������������ʱ�뷽��
		ת��b���õ��ģ����ȡ��ֵ�������ƽ���ı�����������a��˳ʱ�뷽��ת�����õ��ģ����ȡ��ֵ��
		���´��뼴�����ָĳ���ʱ��

		0----1
		 \
		   \
		3    2


*/


		//�������������࣬����ʱ��˳������  
		if (o < 0.0)
		{
			//����swap���������ϣ�http://blog.csdn.net/dizuo/article/details/6435847
			//���������λ��
			swap(marker.points[1], marker.points[3]);
		}
	}

	//�Ƴ�����̫�ӽ���Ԫ��  
	//��һ�μ�������� 
	std::vector< std::pair<int, int> > tooNearCandidates;
	for (size_t i = 0;i<possibleMarkers.size();i++)
	{
		const Marker& m1 = possibleMarkers[i];

		//����ÿһ�������ƽ�����뵽�����һ�������������Ǻ�ѡ  
		//��������maker�ı���֮��ľ��룬�����֮�����͵�ƽ��ֵ����ƽ��ֵ��С������Ϊ����maker�����  

		for (size_t j = i + 1;j<possibleMarkers.size();j++)
		{
			const Marker& m2 = possibleMarkers[j];

			float distSquared = 0;

			for (int c = 0;c<4;c++)
			{
				Point v = m1.points[c] - m2.points[c];
				//�����ĵ�� ---> ����ľ���
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
	//����������������marker�ڲ����ĸ���ľ���ͣ�������ͽ�С�ģ���removlaMask������ǣ�
	//������Ϊ���յ�detectedMarkers  
	std::vector<bool> removalMask(possibleMarkers.size(), false);

	for (size_t i = 0;i<tooNearCandidates.size();i++)
	{	//�ܳ�
		float p1 = perimeter(possibleMarkers[tooNearCandidates[i].first].points);
		float p2 = perimeter(possibleMarkers[tooNearCandidates[i].second].points);

		size_t removalIndex;
		if (p1 > p2)
			removalIndex = tooNearCandidates[i].second;
		else
			removalIndex = tooNearCandidates[i].first;

		removalMask[removalIndex] = true;
	}

	//���ؿ��ܵĶ���  
	detectedMarkers.clear();
	for (size_t i = 0;i<possibleMarkers.size();i++)
	{
		if (!removalMask[i])
			detectedMarkers.push_back(possibleMarkers[i]);/*detectedMarkersΪ����*/
	}
}



//��05��
void MarkerDetector::detectMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers, const vector<Point2f> m_markerCorners2d)
{
	Mat canonicalMarker;
	char name[50] = "";

	vector<Marker> goodMarkers;

	//ʶ����
	for (size_t i = 0;i<detectedMarkers.size();i++)
	{
		Marker& marker = detectedMarkers[i];
		// Find the perspective transfomation that brings current marker to rectangular form
		// �õ���ǰmarker��͸�ӱ任����M
		//�����ȸ����ĸ���Ӧ�ĵ��ҵ�͸�ӱ任����һ�������Ǳ�ǵ����꣬�ڶ����������α��ͼ������ꡣ
		//����ı任����ѱ��ת���ɷ��Σ��Ӷ��������Ƿ�����
		// �ҵ�͸��ͶӰ�����ѱ��ת���ɾ���  
		//����ͼ���ı��ζ�������  
		//���ͼ�����Ӧ���ı��ζ������� 
		Mat M = getPerspectiveTransform(marker.points, m_markerCorners2d);


		// Transform image to get a canonical marker image
		// ����ǰ��marker�任Ϊ����ͶӰ
		//�����ͼ��  
		//�����ͼ��  
		//3x3�任���� 
		warpPerspective(grayscale, canonicalMarker, M, Size(100, 100));
		
		//imshow("��Ƶ2", canonicalMarker);//�Կ�Ч��,С����ԭͼ

		//���ǹ̶���ֵ����  
		//����ͼ��image����Ϊһ��2ֵ��ͨ��ͼ��  
		//�����������飬ÿһ��������һ��point���͵�vector��ʾ   
		//��ֵmax_value ʹ�� CV_THRESH_BINARY �� CV_THRESH_BINARY_INV �����ֵ   
		cv::threshold(canonicalMarker, canonicalMarker, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

		//sprintf(name,"C:\\Users\\Mr.HH\\Desktop\\maker\\warp_%d.jpg",i);
		imshow("��ά��", canonicalMarker);

		int nRotations;
		int id = Marker::getMarkerId(canonicalMarker, nRotations);//5*5 mask,3 bitsУ��,2 bits ����,ÿ��stripe��4������,����5��stripe,����4^5��ID,�׿�Ϊ1���ڿ�Ϊ0
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
		std::vector<Point2f> preciseCorners(4 * goodMarkers.size());//ÿ��marker�ĸ���

		for (size_t i = 0; i<goodMarkers.size(); i++)
		{
			Marker& marker = goodMarkers[i];

			for (int c = 0;c<4;c++)
			{
				preciseCorners[i * 4 + c] = marker.points[c];	//i��ʾ�ڼ���marker��c��ʾĳ��marker�ĵڼ�����
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

	//����ϸ����ľ���ͼƬ
	//�½�һ��ͼ���������궨��С����С��������ͼ��
	cv::Mat markerCornersMat(grayscale.size(), grayscale.type());
	markerCornersMat = cv::Scalar(0);//��ʼ��Ϊ0

	for (size_t i=0; i<goodMarkers.size(); i++)
	{
	goodMarkers[i].drawContour(markerCornersMat, cv::Scalar(255));
	}

	imshow("refine",markerCornersMat);

	detectedMarkers = goodMarkers;

}



//��06��
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