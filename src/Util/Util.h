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
#include <cstdio>
#include <cstdlib>
#include <complex>

#include <libpq-fe.h>

// this custom comparator is defined because Point_<double> does not have a comparator
// comparator is needed for sorting point vectors
struct point_comparator
{
    inline bool operator() (const cv::Point_<double>& p1, const cv::Point_<double>& p2)
    {
        // prioritize sorting by x
        return (p1.x != p2.x) ? (p1.x < p2.x) : (p1.y < p2.y);
    }
};

class Util {
public:
    constexpr static const double k_red =  0.2125;
    constexpr static const double k_green =  0.7154;
    constexpr static const double k_blue = 0.0721;
    constexpr static const double pi = 3.14159265358979323846;
    cv::Mat globalImage;

    Util();
    Util(cv::Mat& image);

    //finds the five points of interest (requires sobel image)
    void findBounds(cv::Mat image, std::vector<cv::Point>& topCurve, std::vector<cv::Point>& bottomCurve);
    //finds the top and bottom curves
    void findFocusPoints(cv::Mat image, std::vector<cv::Point>& POIs);
    //finds all ROI from POIs
    void findROIofPOI(cv::Mat image, std::vector<cv::Point> POIs, std::vector<std::vector<cv::Rect>>& ROIs);

    void splitChannels(const cv::Mat &src, cv::Mat &red, cv::Mat &green, cv::Mat &blue, cv::Mat &lum);
    void sobelOperator(const cv::Mat &src, cv::Mat &destination);
    void esf(cv::Mat &roi_src, cv::Mat &roi_edge, std::vector<cv::Point_<double>> &points_vector);

    void performBinning(const std::vector<cv::Point_<double>> &all_points, std::vector<cv::Point_<double>> &binned_points, int bin_interval=4);

    void imageSimilarity(cv::Mat& newImg, cv::Mat& originalImg, double& similarityScore);

    double getDeltaE(cv::Scalar &pixel1, cv::Scalar &pixel2);

    void getMeanColourCIELAB(const cv::Mat &image, cv::Rect &roi, cv::Scalar &mean);

private:

    //helper to findBounds
    void findTopCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsTop);
    void findBottomCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsBottom);
    //constructs a porabola from smaple points (helper to findBounds)
    void findSystemOfEquations(std::vector<cv::Point>& curveSamplePoints, std::vector<cv::Point>& curve, int width);
    void Util::findSystemOfEquationsSideways(std::vector<cv::Point>& curveSamplePoints, std::vector<cv::Point>& curve, int height);
    //helper to findFocusPoints
    void findCorners(cv::Mat& stats, cv::Point& centerOfChart, std::vector<cv::Point>& POIs);
    void findClosestPoint(cv::Mat& stats, cv::Point& center, cv::Point& closestPoint);

    void findRightROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
    void findTopROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
    void findLeftROI(cv::Mat image, cv::Point start, cv::Rect& ROI);
    void findBottomROI(cv::Mat image, cv::Point start, cv::Rect& ROI);


public:
    //takes in a sobel image
    void findCornersOfChart(cv::Mat& image, std::vector<cv::Point> POIs);
    void findLeftLine(cv::Mat image, std::vector<cv::Point>& leftPts);
    void findRightLine(cv::Mat image, std::vector<cv::Point>& rightPts);

    void MSE(cv::Mat img, cv::Mat original, double& nmse);
    //finding coefficients of the ETF
//    void deriveETF(std::vector<float>& coeff);
//    //polynomial FFT
//    void polynomialFFT(std::vector<float>& polynomial);
//    void fft(std::vector<std::complex<double>>& values);


    std::string type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth ) {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }

};


#endif //UTIL_UTIL_H
