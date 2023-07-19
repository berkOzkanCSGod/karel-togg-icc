#include "Util.h"

Util::Util(){

}

Util::Util(cv::Mat& image){
    globalImage = image;
}

void Util::findBounds(cv::Mat image, std::vector<cv::Point>& topCurve, std::vector<cv::Point>& bottomCurve){

    cv::Mat gaussianBlueImage;
    cv::Mat thresholdImage;
    cv::Mat hsvImg;
    std::vector<cv::Point> topCurveSamplePoints;
    std::vector<cv::Point> bottomCurveSamplePoints;

    cv::GaussianBlur(image,gaussianBlueImage,cv::Size(9,3), 1, 0);
    cv::threshold(gaussianBlueImage, thresholdImage, 120, 255, cv::THRESH_BINARY);
    cv::cvtColor(thresholdImage, hsvImg, cv::COLOR_BGR2HSV);
    
    std::thread checkingTop(&Util::findTopCurveSamplePoint, this, hsvImg, std::ref(topCurveSamplePoints));
    std::thread checkingBottom(&Util::findBottomCurveSamplePoint, this, hsvImg, std::ref(bottomCurveSamplePoints));
    checkingBottom.join();
    checkingTop.join();

    findSystemOfEquations(topCurveSamplePoints, topCurve, image.cols);
    findSystemOfEquations(bottomCurveSamplePoints, bottomCurve, image.cols);

}
void Util::findTopCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsTop){
    for (int x = 0; x < image.cols; x += 55) {
        for (int y = 0; y < image.rows/5; y += 1) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y,x);
            uchar v = pixel[2];
//            uchar h = pixel[0];
//            uchar s = pixel[1];
            if (v != 0) {
                samplePointsTop.push_back(cv::Point(x,y));
                break;
            }
        }
    }
}
void Util::findBottomCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsBottom){
    for (int x = 0; x < image.cols; x += 35) {
        for (int y = image.rows-1; y > image.rows-image.rows/5; y -= 1) {
            cv::Scalar color;
            cv::Vec3b pixel = image.at<cv::Vec3b>(y,x);
            uchar v = pixel[2];
//            uchar h = pixel[0];
//            uchar s = pixel[1];
            if (v != 0) {
                samplePointsBottom.push_back(cv::Point(x,y));
                break;
            }
        }
    }
}
void Util::findSystemOfEquations(std::vector<cv::Point>& curveSamplePoints, std::vector<cv::Point>& curve, int width){

    int n = curveSamplePoints.size();
    cv::Mat A(n, 3, CV_32F);
    cv::Mat B(n, 1, CV_32F);

    for(int i = 0; i < n; ++i)
    {
        cv::Point pt = curveSamplePoints[i];
        A.at<float>(i, 0) = pt.x * pt.x; // x^2
        A.at<float>(i, 1) = pt.x;        // x
        A.at<float>(i, 2) = 1;            // 1
        B.at<float>(i) = pt.y;            // y
    }

    cv::Mat X;
    cv::solve(A, B, X, cv::DECOMP_SVD);

    float a = X.at<float>(0);
    float b = X.at<float>(1);
    float c = X.at<float>(2);

    curve.clear();

    for (int x = 0; x < width; ++x)
    {
        int y = cv::saturate_cast<int>(a * x * x + b * x + c);
        cv::Point curr_point(x, y);
        curve.push_back(curr_point);
    }

}
void Util::findFocusPoints(cv::Mat image, std::vector<cv::Point>& POIs){

    cv::Mat patternMatchedImg;
    cv::Mat thresholdImg;
    cv::Mat scaledImg;
    cv::Mat invertedImg;
    int numOfComponents;
    cv::Mat labels, stats, centroids;
    std::vector<cv::Point> topBound,bottomBound;

    cv::Point boundsMidpoint;
    cv::Point centerOfChart;
    cv::Mat target = cv::imread("Resources\\sobelTarget.jpg");

    cv::matchTemplate(image, target, patternMatchedImg, cv::TM_SQDIFF_NORMED);
    cv::threshold(patternMatchedImg, thresholdImg, 0.99, 1, cv::THRESH_BINARY);

    scaledImg = patternMatchedImg * 255;
    scaledImg.convertTo(scaledImg, CV_8U);

    cv::bitwise_not(scaledImg, invertedImg);

    numOfComponents = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids);
    findBounds(image, topBound, bottomBound);

    cv::Point topExtrema = *std::max_element(topBound.begin(), topBound.end(),
                                            [](const cv::Point& pt1, const cv::Point& pt2){
                                                    return pt1.y > pt2.y;
                                            });
    cv::Point bottomExtrema = *std::max_element(bottomBound.begin(), bottomBound.end(),
                                             [](const cv::Point& pt1, const cv::Point& pt2){
                                                 return pt1.y < pt2.y;
                                             });

    boundsMidpoint.x = (topExtrema.x+bottomExtrema.x)/2;
    boundsMidpoint.y = (topExtrema.y+bottomExtrema.y)/2;

    findClosestPoint(stats, boundsMidpoint, centerOfChart);
    POIs.push_back(centerOfChart);

    findCorners(stats, centerOfChart, POIs);


}
void Util::findClosestPoint(cv::Mat& stats, cv::Point& center, cv::Point& closestPoint) {

    float minDist = -1;
    cv::Point minDistPoint;

    for (int i = 1; i < stats.rows; ++i) {
        // Get the bounding box coordinates for the current component
        int left = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT);

        cv::Point currentPoint(left+(width/2), top+(height/2));

        float dist = cv::norm(currentPoint - center);
        if (dist < minDist || minDist < 0){
            minDist = dist;
            minDistPoint = currentPoint;
        }
    }

    //get sobelTarget.jpg width and height and add shift to find center
    closestPoint = minDistPoint;
    closestPoint.x+=32;
    closestPoint.y+=30;
}
void Util::findCorners(cv::Mat& stats, cv::Point& centerOfChart, std::vector<cv::Point>& POIs) {

    std::vector<cv::Point> allPoints;

    for (int i = 1; i < stats.rows; ++i) {
        // Get the bounding box coordinates for the current component
        int left = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_LEFT);
        int top = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::ConnectedComponentsTypes::CC_STAT_HEIGHT);

        cv::Point currentPoint(left+(width/2), top+(height/2));
        //get sobelTarget.jpg width and height and add shift to find center
        currentPoint.x += 32;
        currentPoint.y += 30;
        allPoints.push_back(currentPoint);
//        std::cout << "X: " << currentPoint.x << " Y: " << currentPoint.y << std::endl;
    }

    std::sort(allPoints.begin(), allPoints.end(), [&](const cv::Point& p1, const cv::Point& p2) {
        auto d1 = std::hypot(p1.x - centerOfChart.x, p1.y - centerOfChart.y);
        auto d2 = std::hypot(p2.x - centerOfChart.x, p2.y - centerOfChart.y);
        return d1 > d2;
    });

    for (int i = 0; i < 4; ++i) {
        POIs.push_back(allPoints[i]);
    }

}

void Util::findROIofPOI(cv::Mat image, std::vector<cv::Point> POIs, std::vector<std::vector<cv::Rect>>& ROIs){
    int ROIwidth = 10;
    int ROIheight = 20;
    float startX, startY; //initial point to start searching
    int offset = 30; //offset from startX and Y because we want to look for line and not center figure
    bool boundHit;
    std::vector<cv::Rect> tempRectList;
    cv::Mat convertedImg;
    cv::Mat maskImg;


    if (image.channels() > 1){
        cv::cvtColor(image, convertedImg, CV_8UC1);
    } else {
        convertedImg = image;
    }

    for(auto POI : POIs){
        for (int directionIndex = 0; directionIndex < 4; directionIndex++) {
            startX = POI.x;
            startY = POI.y;
            boundHit = false;
            cv::Rect tempRect(0,0, ROIwidth, ROIheight);
            switch (directionIndex) {
                case 0: //right
                    findRightROI(image, cv::Point(startX + offset, startY), tempRect);
                    break;
                case 1: //left
                    findLeftROI(image, cv::Point(startX - offset, startY), tempRect);
                    break;
                case 2: //top
                    findTopROI(image, cv::Point(startX, startY - offset), tempRect);
                    break;
                case 3: //bottom
                    findBottomROI(image, cv::Point(startX, startY + offset), tempRect);
                    break;
                default:
                    std::cout << "ERROR WHILE LOOKING FOR BOUNDS!!!!!\n";
                    boundHit = true;
                    break;
            }
            tempRect.x -= ROIwidth/2;
            tempRect.y -= ROIheight/2;
            tempRectList.push_back(tempRect);
        }
        ROIs.push_back(tempRectList);
        tempRectList.clear();
    }
}

void Util::findRightROI(cv::Mat image, cv::Point start, cv::Rect& ROI){
    bool boundHit = false;
    cv::Point initial(start);
    cv::Mat binaryImg;

    cv::threshold(image, binaryImg, 30,255,cv::THRESH_BINARY);
    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

    while(!boundHit){

        if (binaryImg.at<uchar>(start.y, start.x) != 0){
            boundHit = true;
            break;
        } else {
            start.x += 1;
        }

        if (start.x > image.cols){
            !boundHit;
            break;
        }
    }

    ROI.x = start.x;
    ROI.y = start.y;
}
void Util::findTopROI(cv::Mat image, cv::Point start, cv::Rect& ROI){
    bool boundHit = false;
    cv::Point initial(start);
    cv::Mat binaryImg;

    cv::threshold(image, binaryImg, 30,255,cv::THRESH_BINARY);
    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

    while(!boundHit){

        if (binaryImg.at<uchar>(start.y, start.x) != 0){
            boundHit = true;
            break;
        } else {
            start.y -= 1;
        }

        if (start.y < 0){
            !boundHit;
            break;
        }
    }

    ROI.x = start.x;
    ROI.y = start.y;
}
void Util::findLeftROI(cv::Mat image, cv::Point start, cv::Rect& ROI){
    bool boundHit = false;
    cv::Point initial(start);
    cv::Mat binaryImg;

    cv::threshold(image, binaryImg, 30,255,cv::THRESH_BINARY);
    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

    while(!boundHit){

        if (binaryImg.at<uchar>(start.y, start.x) != 0){
            boundHit = true;
            break;
        } else {
            start.x -= 1;
        }

        if (start.x < 0){
            !boundHit;
            break;
        }
    }

    ROI.x = start.x;
    ROI.y = start.y;
}
void Util::findBottomROI(cv::Mat image, cv::Point start, cv::Rect& ROI){
    bool boundHit = false;
    cv::Point initial(start);
    cv::Mat binaryImg;

    cv::threshold(image, binaryImg, 30,255,cv::THRESH_BINARY);
    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

    while(!boundHit){

        if (binaryImg.at<uchar>(start.y, start.x) != 0){
            boundHit = true;
            break;
        } else {
            start.y += 1;
        }

        if (start.y > image.rows){
            !boundHit;
            break;
        }
    }

    ROI.x = start.x;
    ROI.y = start.y;
}

void Util::split_channels(const cv::Mat &src, cv::Mat &red, cv::Mat &green, cv::Mat &blue, cv::Mat &lum) {
    cv::Mat channels[3];
    split(src, channels);
    blue = channels[0];
    green = channels[1];
    red = channels[2];
    lum = k_red * red + k_green * green + k_blue * blue;
}

// dst is in HSV colour format
void Util::sobel_operator(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat kernelH, kernelV, resH, resV;
    kernelV = (cv::Mat_<int>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    transpose(kernelV, kernelH);

    filter2D(src, resH, CV_64F, kernelH);
    filter2D(src, resV, CV_64F, kernelV);
    magnitude(resH, resV, dst);
    normalize(dst, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat dir = cv::Mat(dst.size(), CV_64FC1);
    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            dir.at<double>(i, j) = atan2(resV.at<double>(i, j), resH.at<double>(i, j));
        }
    }
    normalize(dir, dir, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    std::vector<cv::Mat> channels(3);
    channels[0] = dir;
    channels[1] = cv::Mat(dst.size(), CV_8UC1, cv::Scalar(255));
    channels[2] = dst;
    merge(channels, dst);
}

// roi_src is expected to be grayscale
// roi_edge is expected to be HSV. H represents the edge angel, S is constant and V represents edge amplitude
void Util::esf(cv::Mat &roi_src, cv::Mat &roi_edge, std::vector<cv::Point_<double>> &points_vector) {

    if (roi_edge.type() != CV_8UC3) {
        std::cout << "Invalid ROI Matrix type!" << std::endl;
        return;
    }

    std::vector<cv::Mat> channels;

    split(roi_edge, channels);

    cv::Mat roi_thresh;
    threshold(channels[2], roi_thresh, 120, 255, cv::THRESH_TOZERO);

    double m_sum = 0;
    double m_count = 0;
    for (int i = 0; i < roi_thresh.rows; ++i) {
        for (int j = 0; j < roi_thresh.cols; ++j) {
            if (roi_thresh.at<uchar>(i, j) != 0) {
                m_count++;
                m_sum += channels[0].at<uchar>(i, j);
            }
        }
    }
    double m_ave = m_sum / m_count;

    std::unordered_map<double, double> points_map;
    for (int i = 0; i < roi_thresh.rows; i++) {
        for (int j = 0; j < roi_thresh.cols; j++) {
            double x = (double) i - ((double) j/m_ave);
            if (points_map.find(x) == points_map.end()) {
                points_map[x] = roi_src.at<uchar>(i, j);
            }
        }
    }
    for (auto pt: points_map) {
        points_vector.emplace_back(pt.first, pt.second);
    }
}


