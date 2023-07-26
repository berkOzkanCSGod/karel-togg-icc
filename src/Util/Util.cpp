#include "Util.h"


Util::Util(){

}

Util::Util(cv::Mat& image){
    globalImage = image;
}

//takes in a sobel image. If input it hsv channel #2 can be sent
void Util::findBounds(cv::Mat image, std::vector<cv::Point>& topCurve, std::vector<cv::Point>& bottomCurve){

    cv::Mat gaussianBlurImage;
    cv::Mat thresholdImage;
    cv::Mat hsvImg;
    std::vector<cv::Point> topCurveSamplePoints;
    std::vector<cv::Point> bottomCurveSamplePoints;

    cv::GaussianBlur(image,gaussianBlurImage,cv::Size(9,3), 1, 0);
    cv::threshold(gaussianBlurImage, thresholdImage, 120, 255, cv::THRESH_BINARY);
//    cv::cvtColor(thresholdImage, hsvImg, cv::COLOR_BGR2HSV);
    
    std::thread checkingTop(&Util::findTopCurveSamplePoint, this, thresholdImage, std::ref(topCurveSamplePoints));
    std::thread checkingBottom(&Util::findBottomCurveSamplePoint, this, thresholdImage, std::ref(bottomCurveSamplePoints));
    checkingBottom.join();
    checkingTop.join();

    findSystemOfEquations(topCurveSamplePoints, topCurve, image.cols);
    findSystemOfEquations(bottomCurveSamplePoints, bottomCurve, image.cols);

}
//takes in thresh binary image
void Util::findTopCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsTop){
    for (int x = 0; x < image.cols; x += 55) {
        for (int y = 0; y < image.rows/5; y += 1) {
            uchar pixel = image.at<uchar>(y,x);
//            uchar v = pixel[2];
//            uchar h = pixel[0];
//            uchar s = pixel[1];
            if (pixel != 0) {
                samplePointsTop.push_back(cv::Point(x,y));
                break;
            }
        }
    }
}
//takes in thresh binary image
void Util::findBottomCurveSamplePoint(cv::Mat image, std::vector<cv::Point>& samplePointsBottom){
    for (int x = 0; x < image.cols; x += 35) {
        for (int y = image.rows-1; y > image.rows-image.rows/5; y -= 1) {
            cv::Scalar color;
            uchar pixel = image.at<uchar>(y,x);
//            uchar v = pixel[2];
//            uchar h = pixel[0];
//            uchar s = pixel[1];
            if (pixel != 0) {
                samplePointsBottom.push_back(cv::Point(x,y));
                break;
            }
        }
    }
}
//takes in two vectors of points
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

//takes in a sobel image (black and white)
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
    cv::Mat target = cv::imread("Resources\\sobelTarget.jpg", cv::IMREAD_GRAYSCALE);
    //maybe get target some other way (improvement)

    image.convertTo(image, CV_8U);

    cv::Mat blur;
    cv::matchTemplate(image, target, patternMatchedImg, cv::TM_SQDIFF_NORMED);

    cv::threshold(patternMatchedImg, thresholdImg, 0.9999, 1, cv::THRESH_BINARY);
    cv::GaussianBlur(thresholdImg,blur,cv::Size(9,9),0,0);
    cv::threshold(blur, thresholdImg, 0.9, 1, cv::THRESH_BINARY);

    scaledImg = thresholdImg * 255;
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

//    cv::cvtColor(thresholdImg, thresholdImg, cv::COLOR_GRAY2BGR);
//
//    for(int i = 1; i < numOfComponents; i++)
//    {
//        cv::Point center(cvRound(centroids.at<double>(i, 0)), cvRound(centroids.at<double>(i, 1)));
//        cv::circle(thresholdImg, center, 3, cv::Scalar(0, 0, 255), -1);
//    }
//    cv::imshow("final", thresholdImg);
//    cv::waitKey(0);


}
//takes in a matrix of stats produced by "connectedComponentsWithStats" and center of chart
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
    for(auto& p : POIs){
        cv::circle(globalImage, p, 10, cv::Scalar(12,100,50),1);
    }
//    cv::imshow("global", globalImage);
//    cv::waitKey(0);
}

//takes in sobel image
void Util::findROIofPOI(cv::Mat image, std::vector<cv::Point> POIs, std::vector<std::vector<cv::Rect>>& ROIs){
    int ROIwidth = 10;
    int ROIheight = 20;
    float startX, startY; //initial point to start searching
    int offset = 30; //offset from startX and Y because we want to look for line and not center figure
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
            cv::Rect tempRect(0,0, ROIwidth, ROIheight);
            switch (directionIndex) {
                case 0: //right
                    findRightROI(convertedImg, cv::Point(startX + offset, startY), tempRect);
                    break;
                case 1: //left
                    findLeftROI(convertedImg, cv::Point(startX - offset, startY), tempRect);
                    break;
                case 2: //top
                    findTopROI(convertedImg, cv::Point(startX, startY - offset), tempRect);
                    break;
                case 3: //bottom
                    findBottomROI(convertedImg, cv::Point(startX, startY + offset), tempRect);
                    break;
                default:
                    std::cout << "ERROR WHILE LOOKING FOR BOUNDS!!!!!\n";
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
//takes in sobel img
void Util::findRightROI(cv::Mat image, cv::Point start, cv::Rect& ROI){
    bool boundHit = false;
    cv::Point initial(start);
    cv::Mat binaryImg;

    cv::threshold(image, binaryImg, 30,255,cv::THRESH_BINARY);

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
//    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

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
//    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

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
//    cv::cvtColor(binaryImg, binaryImg, cv::COLOR_BGR2GRAY);

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

/*
 * Performs channel splitting operation on a src image, and outputs the channels + luminance as image matrix.
 */
void Util::splitChannels(const cv::Mat &src, cv::Mat &red, cv::Mat &green, cv::Mat &blue, cv::Mat &lum) {
    cv::Mat channels[3];
    cv::split(src, channels);
    blue = channels[0];
    green = channels[1];
    red = channels[2];
    lum = k_red * red + k_green * green + k_blue * blue;
}

/*
 * Takes in a gray scale src image and outputs an hsv image (destination)
 */
void Util::sobelOperator(const cv::Mat &src, cv::Mat &destination) {

    //check if gray, if not make it gray
    if (src.channels() != 1) {
        cv::cvtColor(src, src, CV_8UC1);
    }

    cv::Mat kernelHue, kernelVal, resHue, resVal;
    cv::Mat direction;
    std::vector<cv::Mat> channels(3);

    kernelVal = (cv::Mat_<int>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    cv::transpose(kernelVal, kernelHue); //rotates kernelVal 90 deg and assigns it on to kernelHue.

    cv::filter2D(src, resHue, CV_64F, kernelHue);
    cv::filter2D(src, resVal, CV_64F, kernelVal);
    cv::magnitude(resHue, resVal, destination); //calculates 2d vector magnitude

    //destination is gray scale
    cv::normalize(destination, destination, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    direction = cv::Mat(destination.size(), CV_64FC1);
    for (int i = 0; i < destination.rows; ++i) {
        for (int j = 0; j < destination.cols; ++j) {
            direction.at<double>(i, j) = atan2(resVal.at<double>(i, j), resHue.at<double>(i, j));
        }
    }

    //direction is gray scale
    cv::normalize(direction, direction, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    channels[0] = direction; //gray scale
    channels[1] = cv::Mat(destination.size(), CV_8UC1, cv::Scalar(255)); //completely white grayscale
    channels[2] = destination; //gray scale

    cv::merge(channels, destination);
}

// roi_src is expected to be grayscale
// roi_edge is expected to be HSV. H represents the edge angel, S is constant and V represents edge amplitude
void Util::esf(cv::Mat &roiSrc, cv::Mat &roiEdge, std::vector<cv::Point_<double>> &pointsVector) {
    std::vector<cv::Mat> channels;
    cv::Mat roiHue;
    cv::Mat roiVal;
    cv::Mat threshROI;
    std::vector<int> row1val;
    std::vector<int> rowLastval;
    double sum1, sum2, avg1, avg2, slope;
    double x0,y0,x1,y1;



    if (roiEdge.type() != CV_8UC3) {
        std::cout << "Invalid ROI Matrix type!" << std::endl;
        return;
    }


    split(roiEdge, channels);
    roiHue = channels[0];
    roiVal = channels[2];
    cv::threshold(roiVal, threshROI, 100, 255, cv::THRESH_BINARY);

    //first row
    for (int i = 0; i < threshROI.cols; ++i) {
        if (threshROI.at<uchar>(0,i) != 0){
            row1val.push_back(i);
        }
    }
    sum1 = std::accumulate(row1val.begin(), row1val.end(), 0.0);
    avg1 = sum1/(double)row1val.size();
    //last row
    for (int i = 0; i < threshROI.cols; ++i) {
        if (threshROI.at<uchar>(threshROI.rows-1,i) != 0){
            rowLastval.push_back(i);
        }
    }
    sum2 = std::accumulate(rowLastval.begin(), rowLastval.end(), 0.0);
    avg2 = sum2/(double)rowLastval.size();

 // based on:
 //      y1-y0
 // m = --------
 //      x1-x0

    x0 = avg1;
    y0 = 0;
    x1 = avg2;
    y1 = threshROI.rows;
    slope = (-(y0-y1))/(x0-x1); //negative because rows are counted downwards
    for (int yA = 0; yA < roiHue.rows; yA++) {
        for (int xA = 0; xA < roiHue.cols; xA++) {
            double x = (double) xA - ((double) yA / atan(slope));
                pointsVector.emplace_back(cv::Point_<double>(x, roiSrc.at<uchar>(yA, xA)));
        }
    }

    // sort the points according to x values to prepare for binning with a custom comparator
    std::sort(pointsVector.begin(), pointsVector.end(), point_comparator());

//    std::unordered_map<double, double> points_map;
//    for (int yA = 0; yA < roi_hue.rows; yA++) {
//        for (int xA = 0; xA < roi_hue.cols; xA++) {
//            double x = (double) xA - ((double) yA / atan(slope));
//            if (pointsMap.find(x) == points_map.end()) {
//                points_map[x] = roiSrc.at<uchar>(yA, xA);
//            }
//        }
//    }

//        for (auto pt: points_map) {
//            pointsVector.emplace_back(pt.first, pt.second);
//        }

}

// performs point binning in O(N) time complexity. Each bin contains bin_interval points
// bin values are calculated with arithmetical average of points in each bin
void Util::performBinning(const std::vector<cv::Point_<double>> &all_points, std::vector<cv::Point_<double>> &binned_points,
                     int bin_interval) {

    // bin_interval is the number of consecutive points to be binned
    assert(bin_interval > 1);

    // clear binned_points if it contains data
    if (!binned_points.empty()) {
        binned_points.clear();
    }

    // group bins. If the last bin cannot have the desired size, skip it
    for (int i = 0; i < all_points.size() - bin_interval; i+=bin_interval) {
        double x_sum = 0;
        double y_sum = 0;

        // add n = bin_interval points' values together
        for (int j = 0; j < bin_interval; ++j) {
            cv::Point_<double> point = all_points[i + j];
            x_sum += point.x;
            y_sum += point.y;
        }

        // take the average
        cv::Point_<double> ave_point(x_sum / bin_interval, y_sum / bin_interval);
        // append the result to the destination
        binned_points.emplace_back(ave_point);
    }

    // put the remaining n <= bin_interval points into the last bin
    double x_sum = 0;
    double y_sum = 0;
    for (int i = binned_points.size() * bin_interval; i < all_points.size(); ++i) {
        cv::Point_<double> point = all_points[i];
        x_sum += point.x;
        y_sum += point.y;
    }
    cv::Point_<double> ave_point(x_sum/bin_interval, y_sum/bin_interval);
    binned_points.emplace_back(ave_point);

}

//needs to test if images are same size
void Util::imageSimilarity(cv::Mat& newImg, cv::Mat& originalImg, double& similarityScore){
//
//    if (newImg.size() != originalImg.size() || newImg.type() != originalImg.type()) {
//        std::cout << "The images have different sizes or types. Cannot calculate similarity." << std::endl;
//        return;
//    }
//
//    cv::Mat diff;
//    double mse, max_mse, nmse;
//
//    cv::absdiff(newImg, originalImg, diff);
//
//    double sum = 0;
//    for (int i = 0; i < diff.rows; ++i) {
//        for(int j = 0; j < diff.cols; ++j){
//            sum += (double)diff.at<uchar>(i,j)/255;
//            std::cout << (double)diff.at<uchar>(i,j)/255 << "\n";
//        }
//    }
//
//    similarityScore = (sum / (diff.rows*diff.cols));
//
////    diff.convertTo(diff, CV_32F);
////    mse = cv::norm(diff, cv::NORM_L2SQR) / (diff.rows * diff.cols);
////    max_mse = 255.0 * 255.0;
////    nmse = mse / max_mse;
////    similarityScore = 1 - nmse;
}


// finds getDeltaE value from two pixels in CIELAB colour space with the formula:
// getDeltaE = sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)
double Util::getDeltaE(cv::Scalar &pixel1, cv::Scalar &pixel2) {
    return cv::norm(pixel1 - pixel2);
}


// Takes an image in BGR colour space and finds the mean colours in CIELAB colour space
void Util::getMeanColourCIELAB(const cv::Mat &image, cv::Rect &roi, cv::Scalar &mean) {
    // Extract the region of interest (ROI) from the image
    cv::Mat roiImage = image(roi);

    // Calculate the mean colour in BGR colour space
    cv::Scalar meanColour = cv::mean(roiImage);

    // Create a 1x1 BGR cv::Mat from the mean colour
    cv::Mat bgrMat(1, 1, CV_64FC3, meanColour);

    // Convert the mean color from BGR to CIELAB
    cv::Mat labMat;
    cv::cvtColor(bgrMat, labMat, cv::COLOR_BGR2Lab);

    // Extract the individual LAB components from the converted Mat
    mean = labMat.at<cv::Scalar>(0, 0);
}
