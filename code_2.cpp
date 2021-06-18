#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/persistence.hpp>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    string path1 = samples::findFile(argv[1]);
    string path2 = samples::findFile(argv[2]);
    Mat img1 = imread(path1, IMREAD_COLOR);  
    Mat img2 = imread(path2, IMREAD_COLOR);  
    int width1 = img1.cols;
    int height1 = img1.rows;
    int width2 = img2.cols;
    int height2 = img2.rows;
    cv::Rect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = min(width1, width2);
    roi.height = min(height1, height2);
    cout<<" width: "<<roi.width <<", and the height: "<<roi.height<<"\n";
    Mat img_1 = img1(roi);
    Mat img_2 = img2(roi);
    imshow("first",img_1);
    waitKey(0);
    imshow("second",img_2);
    waitKey(0);
    return 0;
}
