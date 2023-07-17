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

class Util {
public:
    cv::Mat globalImage;

    Util();
    Util(cv::Mat& image);

    void findFocusPoints(cv::Mat image, std::vector<cv::Point>& POIs);
    void findBounds(cv::Mat image, std::vector<cv::Point>& topCurve, std::vector<cv::Point>& bottomCurve);

private:
    void findTopCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsTop);
    void findBottomCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsBottom);

    void Util::findSystemOfEquations(std::vector<cv::Point>& curveSamplePoints, std::vector<cv::Point>& curve, int width);
    void findCorners(cv::Mat& stats, cv::Point& centerOfChart, std::vector<cv::Point>& POIs);
    void findClosestPoint(cv::Mat& stats, cv::Point& center, cv::Point& closestPoint);
};


#endif //UTIL_UTIL_H
