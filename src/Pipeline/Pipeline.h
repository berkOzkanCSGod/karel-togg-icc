//
// Created by attut on 25-Jul-23.
//

#ifndef KAREL_TOGG_ICC_PIPELINE_H
#define KAREL_TOGG_ICC_PIPELINE_H

#include <libpq-fe.h>
#include <iostream>
#include <thread>
#include <string>
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

class Pipeline {



public:
    Pipeline();
    Pipeline(cv::Mat& image);

    //tests
    void mtfTest(cv::Mat image, double& mtfScore);
    void matchFrame(cv::Mat compareTo, cv::Mat comparedTo, double& similarity);
    void colorDepth(cv::Mat image, double& depthVal);

    void performTests(cv::Mat& image, double& mtfScore, double& similarityScore, double& depthVal);
    void evaluateTestResults(cv::Mat& image, std::string productId);

    //db
    bool connectToDb();
    bool closeDb();
    bool removeById(std::string productId);
    bool updateById(std::string productId, double mtf);
    bool addNewEntry(std::string productId, double product_mtf);
    bool getAllData(PGresult*& response);

private:

    cv::Mat originalImg;

    //db -----------------------
    std::string dbName = "testDB";
    std::string dbUser = "attut";
    std::string dbPassword = "12345678";
    std::string dbPort = "5433";
    std::string dbconnectionStr = "dbname="+dbName+" user="+dbUser+" password="+dbPassword+" hostaddr=127.0.0.1 port="+dbPort;
    PGconn* connection;

    bool executeCmd(const char* SQLcmd, PGresult*& response);
    //db -----------------------
};


#endif //KAREL_TOGG_ICC_PIPELINE_H
