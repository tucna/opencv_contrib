/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2015, University of Ostrava, Institute for Research and Applications of Fuzzy Modeling,
// Pavel Vlasanek, all rights reserved. Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#include <string>

using namespace std;
using namespace cv;

TEST(fuzzy_f0, components)
{
    float arI[16][16] =
    {
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255},
        {0, 0, 0, 10, 34, 57, 80, 104, 127, 150, 174, 197, 221, 244, 255, 255}
    };    
    Mat I = Mat(16, 16, CV_32F, arI);

    float arDemandedComp[9][9] =
    {
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255},
        {0, 2.5, 33.75, 80.25, 127, 173.75, 220.75, 252.25, 255}
    };
    Mat demandedComp = Mat(9, 9, CV_32F, arDemandedComp);

    Mat kernel;
    ft::createKernel(ft::LINEAR, 2, kernel);

    Mat f0comp;
    ft::FT02D_components(I, kernel, f0comp);

    float n1 = cvtest::norm(demandedComp, f0comp, NORM_INF);

    EXPECT_FLOAT_EQ(n1, 0);
}
