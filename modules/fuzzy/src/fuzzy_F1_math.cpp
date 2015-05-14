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

void ft::FT12D_components(InputArray matrix, InputArray kernel, OutputArray components)
{

}

void ft::FT12D_polynomial(InputArray matrix, InputArray kernel, OutputArray c00, OutputArray c10, OutputArray c01, OutputArray components, InputArray mask)
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
    components.create(Bn * kernelMat.rows, An * kernelMat.cols, CV_32F);

    Mat c00Mat = c00.getMat();
    Mat c10Mat = c10.getMat();
    Mat c01Mat = c01.getMat();
    Mat componentsMat = components.getMat();

    Mat vecX;
    Mat vecY;

    FT12D_createPolynomMatrixVertical(radiusX, vecX);
    FT12D_createPolynomMatrixHorizontal(radiusY, vecY);

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

            Mat numerator00, numerator10, numerator01;
            multiply(roiImage, kernelMasked, numerator00, 1, CV_32F);
            multiply(numerator00, vecX, numerator10, 1, CV_32F);
            multiply(numerator00, vecY, numerator01, 1, CV_32F);

            Mat denominator00, denominator10, denominator01;
            denominator00 = kernelMasked;
            multiply(vecX.mul(vecX), kernelMasked, denominator10, 1, CV_32F);
            multiply(vecY.mul(vecY), kernelMasked, denominator01, 1, CV_32F);

            c00Mat.row(o).col(i) = sum(numerator00) / sum(denominator00);
            c10Mat.row(o).col(i) = sum(numerator10) / sum(denominator10);
            c01Mat.row(o).col(i) = sum(numerator01) / sum(denominator01);

            Mat component1(componentsMat, Rect(i * kernelMat.cols, o * kernelMat.rows, kernelMat.cols, kernelMat.rows));

            //component1 =  c10Mat.at<float>(o,i) * vecX;// + c01Mat.at<float>(o,i) * vecY;

            Mat updatedC10;
            Mat updatedC01;

            multiply(c10Mat.at<float>(o,i), vecX, updatedC10, 1, CV_32F);
            multiply(c01Mat.at<float>(o,i), vecY, updatedC01, 1, CV_32F);

            add(updatedC01, updatedC10, component1);
            add(component1, c00Mat.at<float>(o,i), component1);
        }
    }
}

void ft::FT12D_createPolynomMatrixVertical(int radius, OutputArray matrix)
{
    int dimension = radius * 2 + 1;

    matrix.create(dimension, dimension, CV_16S);

    Mat matrixMat = matrix.getMat();

    matrixMat = 0;

    for (int i = 0; i < radius; i++)
    {
        matrixMat.col(i) = i - radius;
        matrixMat.col(dimension - 1 - i) = radius - i;
    }
}

void ft::FT12D_createPolynomMatrixHorizontal(int radius, OutputArray matrix)
{
    int dimension = radius * 2 + 1;

    matrix.create(dimension, dimension, CV_16S);

    Mat matrixMat = matrix.getMat();

    matrixMat = 0;

    for (int i = 0; i < radius; i++)
    {
        matrixMat.row(i) = i - radius;
        matrixMat.row(dimension - 1 - i) = radius - i;
    }
}
