#pragma once
#include "iostream"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "Marker.h"
/**
* A top-level class that encapsulate marker detector algorithm
*/


using namespace cv;
using namespace std;


class MarkerDetector
{
public:
	/**
	* Initialize a new instance of marker detector object
	* @calibration[in] - Camera calibration necessary for pose
	estimation.
	*/
	//MarkerDetector(CameraCalibration calibration);

	//void processFrame(const BGRAVideoFrame& frame);

	//const vector<Transformation>& getTransformations() const;

	//protected:

	//bool findMarkers(const BGRAVideoFrame& frame, vector<Marker>& detectedMarkers);

	void prepareImage(const Mat &bgraMat,Mat &grayscale);

	void performThreshold(const Mat &grayscale,Mat &thresholdImg);

	void findContours(const Mat& thresholdImg,vector<vector<Point> >& contours,int minContourPointsAllowed);

	void findMarkerCandidates(const vector<vector<Point> >& contours, vector<Marker>& detectedMarkers);

	void detectMarkers(const Mat& grayscale, vector<Marker>& detectedMarkers, const vector<Point2f> m_markerCorners2d);

	void estimatePosition(vector<Marker>& detectedMarkers, const vector<Point3f> &m_markerCorners3d, const Matrix33 m_intrinsic, const Vector4  m_distorsion);


};
