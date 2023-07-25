#include "Pipeline.h"


Pipeline::Pipeline(){

}
Pipeline::Pipeline(cv::Mat& image){

}

void Pipeline::mtfTest(cv::Mat image, double& mtfScore){
    //do mtf test here and output a value between 0 and 1.
}

void Pipeline::matchFrame(cv::Mat compareTo, cv::Mat comparedTo, double& similarity){
    //finds the difference between two frames and outputes a similarity score between 0 and 1,
    //based on a thresold
}

void Pipeline::colorDepth(cv::Mat image, double& depthVal){
    //outputs the calculated depth value
}

void Pipeline::performTests(cv::Mat& image, double& mtfScore, double& similarityScore, double& depthVal){

    std::thread mtfTester(&Pipeline::mtfTest, this, image,std::ref(mtfScore));
    std::thread matchFrameTester(&Pipeline::matchFrame, this, image, originalImg, std::ref(similarityScore));
    std::thread colorTester(&Pipeline::colorDepth, this, image, std::ref(depthVal));

    mtfTester.join();
    matchFrameTester.join();
    colorTester.join();

}

void Pipeline::evaluateTestResults(cv::Mat& image, std::string productId){
    double mtfScore, similarityScore, depthVal;
    double mtfBenchmark = 0.8, similarityBenchmark = 0.9, colorBenchmark = 0.5;
    bool mtfTest = false, similarityTest = false, colorTest = false;

    performTests(image, mtfScore, similarityScore, depthVal);

    if (mtfScore <= 1 && mtfScore >= mtfBenchmark)
        mtfTest = true;
    if (similarityScore <= 1 && similarityScore >= similarityBenchmark)
        similarityScore = true;
    if (depthVal <= 1 && depthVal >= colorBenchmark)
        colorTest = true;

    addNewEntry(productId, mtfScore); //this should obviously change with db changes
}


//db
bool Pipeline::connectToDb(){
    connection = PQconnectdb(dbconnectionStr.data());
    if (PQstatus(connection) != CONNECTION_OK){
        return false;
    }

    return true;
}
bool Pipeline::executeCmd(const char* SQLcmd, PGresult*& response){

    if (PQstatus(connection) == CONNECTION_OK){
        response = PQexec(connection, SQLcmd);

        if (PQresultStatus(response) != PGRES_TUPLES_OK){
            PQclear(response);
            return false;
        } else {
            return true;
        }
    } else {
        return false;
    }
}
bool Pipeline::addNewEntry(std::string productId, double product_mtf){
    if (PQstatus(connection) != CONNECTION_OK){
        return false;
    }

    std::string cmd = "INSERT INTO public.product_history (product_id,mtf) VALUES("+productId+","+std::to_string(product_mtf)+");";
    PGresult* res;

    return executeCmd(cmd.data(),res);

}
bool Pipeline::getAllData(PGresult*& response){
    if (PQstatus(connection) != CONNECTION_OK){
        return false;
    }

    std::string cmd = "SELECT * FROM public.product_history";

    return executeCmd(cmd.data(),response);
}
bool Pipeline::closeDb(){
    if (PQstatus(connection) == CONNECTION_OK){
        PQfinish(connection);
    }
    return true;
}
bool Pipeline::removeById(std::string productId){
    std::string cmd = "DELETE FROM public.product_history WHERE product_id=\'"+productId+"\';";
    PGresult* res;
    return executeCmd(cmd.data(),res);
}
bool Pipeline::updateById(std::string productId, double mtf){
    std::string cmd = "UPDATE public.product_history SET mtf="+std::to_string(mtf)+" WHERE product_id=\'"+productId+"\';";
    PGresult* res;
    return executeCmd(cmd.data(),res);

}

