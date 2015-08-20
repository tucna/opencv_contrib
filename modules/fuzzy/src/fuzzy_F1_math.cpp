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

#include "precomp.hpp"

using namespace cv;

void ft::FT12D_components(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray components)
{

}

void ft::FT12D_polynomial(cv::InputArray matrix, cv::InputArray kernel, cv::OutputArray c00, cv::OutputArray c10, cv::OutputArray c01)
{
    Mat matrixMat = matrix.getMat();
    Mat kernelMat = kernel.getMat();
    Mat maskMat = mask.getMat();

    CV_Assert(matrixMat.channels() == 1 && kernelMat.channels() == 1 && maskMat.channels() == 1);

    int radiusX = (kernelMat.cols - 1) / 2;
    int radiusY = (kernelMat.rows - 1) / 2;
    int An = matrixMat.cols / radiusX + 1;
    int Bn = matrixMat.rows / radiusY + 1;

    Mat matrixPadded;
    Mat maskPadded;

    copyMakeBorder(matrixMat, matrixPadded, radiusY, kernelMat.rows, radiusX, kernelMat.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(maskMat, maskPadded, radiusY, kernelMat.rows, radiusX, kernelMat.cols, BORDER_CONSTANT, Scalar(0));

    c00.create(Bn, An, CV_32F);
    c10.create(Bn, An, CV_32F);
    c01.create(Bn, An, CV_32F);

    Mat c00Mat = c00.getMat();
    Mat c10Mat = c10.getMat();
    Mat c01Mat = c01.getMat();

    Mat vecX(area.height, area.width, CV_8U);
    Mat vecY(area.height, area.width, CV_8U);

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernelMat.cols, kernelMat.rows);

            Mat roiImage(matrixPadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernelMat.copyTo(kernelMasked, roiMask);

            Mat numerator;
            multiply(roiImage, kernelMasked, numerator, 1, CV_32F);

            c00Mat.row(o).col(i) = sum(numerator) / sum(kernelMasked);
        }
    }
}

