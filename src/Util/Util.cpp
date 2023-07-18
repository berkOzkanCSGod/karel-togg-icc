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

void Util::deriveETF(std::vector<float>& coeff){
    for (int i = 0; i < coeff.size(); ++i) {
        coeff[i]*= i;
    }
    std::rotate(coeff.begin(), coeff.begin()+1, coeff.end());
    coeff.resize(coeff.size() - 1);
}

void Util::polynomialFFT(std::vector<float>& polynomial){
    std::vector<std::complex<double>> coeff(polynomial.size());
    for (int i = 0; i < polynomial.size(); ++i) {
        coeff[i] = polynomial[i];
    }

    fft(coeff);
}
void Util::fft(std::vector<std::complex<double>>& values) {
    const size_t N = values.size();
    if (N <= 1) {
        return;
    }

    std::vector<std::complex<double>> even(N / 2);
    std::vector<std::complex<double>> odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = values[2 * i];
        odd[i] = values[2 * i + 1];
    }

    fft(even);
    fft(odd);

    for (size_t k = 0; k < N / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2 * std::acos(-1.0) * k / N) * odd[k];
        values[k] = even[k] + t;
        values[k + N / 2] = even[k] - t;
    }
}

