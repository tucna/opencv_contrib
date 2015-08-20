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

#ifndef __OPENCV_FUZZY_F1_MATH_H__
#define __OPENCV_FUZZY_F1_MATH_H__

#include "types.hpp"
#include "opencv2/core.hpp"

namespace cv
{

namespace ft
{
    CV_EXPORTS void FT12D_components(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray components);
    CV_EXPORTS void FT12D_polynomial(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray c00, cv::OutputArray c10, cv::OutputArray c01, cv::OutputArray components);
    CV_EXPORTS void FT12D_createPolynomMatrixVertical(int radius, cv::OutputArray matrix, const int chn = 1);
    CV_EXPORTS void FT12D_createPolynomMatrixHorizontal(int radius, cv::OutputArray matrix, const int chn = 1);
    CV_EXPORTS void FT12D_inverseFT(cv::InputArray components, cv::InputArray kernel, cv::OutputArray output, int width, int height);
    CV_EXPORTS void FT12D_process(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask);
    CV_EXPORTS void DUMMY_ft1_inpaint(const cv::Mat &image, const cv::Mat &mask, cv::Mat &output, int radius);
    CV_EXPORTS void FT12D_inverseIrina(const Mat &c01, const Mat &c10, const Mat &kernel, Mat &S10, Mat &S01, Mat &iFT, int width, int height);
}
}

#endif // __OPENCV_FUZZY_F1_MATH_H__
