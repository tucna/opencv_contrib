/* Sample - Filtering
 * Target is to apply filtering using F-transform
 * on the image "input.png". Two different radiuses
 * are used, where bigger one (100 in this case)
 * means higher level of blurriness.
 *
 * Image "output1_filter.png" is created from "input.png"
 * using radius 3.
 *
 * Image "output2_filter.png" is created from "input.png"
 * using radius 100.
 *
 * Both examples use linear basic function (parametr ft:LINEAR).
 */

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/fuzzy.hpp"

using namespace std;
using namespace cv;

int main(void)
{
    // Input image
    Mat I = imread("input.png");

    // Filtering
    Mat output1, output2;

    ft::filter(I, output1, ft::LINEAR, 3);
    ft::filter(I, output2, ft::LINEAR, 3);

    // Save output

    imwrite("output1_filter.png", output1);
    imwrite("output2_filter.png", output2);

    return 0;
}
