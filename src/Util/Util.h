//
// Created by attut on 15-Jul-23.
//

#ifndef UTIL_UTIL_H
#define UTIL_UTIL_H

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <thread>
#include <cmath>
#include <numeric>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

class Util {
public:
    cv::Mat globalImage;

    Util();
    Util(cv::Mat& image);

    //finds the five points of interest (requires sobel image)
    void findBounds(cv::Mat image, std::vector<cv::Point>& topCurve, std::vector<cv::Point>& bottomCurve);
    //finds the top and bottom curves
    void findFocusPoints(cv::Mat image, std::vector<cv::Point>& POIs);
    //finds all ROI from POIs
    void findROIofPOI(cv::Mat image, std::vector<cv::Point> POIs, std::vector<std::vector<cv::Rect>>& ROIs);

private:

    //helper to findBounds
    void findTopCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsTop);
    void findBottomCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsBottom);
    //constructs a porabola from smaple points (helper to findBounds)
    void findSystemOfEquations(std::vector<cv::Point>& curveSamplePoints, std::vector<cv::Point>& curve, int width);

    //helper to findFocusPoints
    void findCorners(cv::Mat& stats, cv::Point& centerOfChart, std::vector<cv::Point>& POIs);
    void findClosestPoint(cv::Mat& stats, cv::Point& center, cv::Point& closestPoint);


    void findRightROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
    void findTopROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
    void findLeftROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
    void findBottomROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
};


#endif //UTIL_UTIL_H
