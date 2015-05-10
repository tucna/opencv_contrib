#include "opencv2/opencv.hpp"
#include "opencv2/fuzzy.hpp"

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
    Mat I = imread("input.png");

    Mat mask1 = imread("mask1.png");
    Mat mask2 = imread("mask2.png");
    Mat mask3 = imread("mask3.png");

    Mat output1, output2, output3, output4;

    ft::inpaint(I, mask1, output1, 2, ft::LINEAR, ft::ONE_STEP);
    ft::inpaint(I, mask2, output2, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(I, mask3, output3, 2, ft::LINEAR, ft::MULTI_STEP);
    ft::inpaint(I, mask3, output4, 2, ft::LINEAR, ft::ITERATIVE);

    imwrite("output1.png", output1);
    imwrite("output2.png", output2);
    imwrite("output3.png", output3);
    imwrite("output4.png", output4);

    return 0;
}
