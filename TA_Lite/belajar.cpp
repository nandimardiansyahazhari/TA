#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
    Mat img = imread("/home/ansyah/Downloads/archive/Cars Dataset/test/Audi/54.jpg", IMREAD_COLOR);

    imshow("/home/ansyah/Downloads/archive/Cars Dataset/test/Audi/54.jpg", img);

    waitKey(0);
}