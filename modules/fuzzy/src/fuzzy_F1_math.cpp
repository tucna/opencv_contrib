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

void ft::FT12D_polynomial(InputArray matrix, InputArray kernel, OutputArray c00, OutputArray c10, OutputArray c01, OutputArray components)
{
    // DIFFERENT BORDERS SO MASK IS USED!!! ERASE
    Mat mask = Mat::ones(matrix.size(), CV_8U);
    //***

    Mat matrixMat = matrix.getMat();
    Mat kernelMat = kernel.getMat();

    CV_Assert(matrixMat.channels() == 1 && kernelMat.channels() == 1);

    int radiusX = (kernelMat.cols - 1) / 2;
    int radiusY = (kernelMat.rows - 1) / 2;
    int An = matrixMat.cols / radiusX + 1;
    int Bn = matrixMat.rows / radiusY + 1;

    Mat matrixPadded;

    copyMakeBorder(matrixMat, matrixPadded, radiusY, kernelMat.rows, radiusX, kernelMat.cols, BORDER_CONSTANT, Scalar(0));

    //***
    Mat maskPadded;

    copyMakeBorder(mask, maskPadded, radiusY, kernelMat.rows, radiusX, kernelMat.cols, BORDER_CONSTANT, Scalar(0));
    //***

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

            //***
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernelMat.copyTo(kernelMasked, roiMask);

            //kernelMat to kernelMasked
            //***

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

            Mat updatedC10;
            Mat updatedC01;

            //***
            Mat vecXMasked, vecYMasked;

            vecX.copyTo(vecXMasked, roiMask);
            vecY.copyTo(vecYMasked, roiMask);

            //vecX to vecXMasked
            //vecY to vecYMasked
            //***

            multiply(c10Mat.at<float>(o,i), vecXMasked, updatedC10, 1, CV_32F);
            multiply(c01Mat.at<float>(o,i), vecYMasked, updatedC01, 1, CV_32F);

            add(updatedC01, updatedC10, component1);
            add(component1, c00Mat.at<float>(o,i), component1);
        }
    }
}

void ft::FT12D_createPolynomMatrixVertical(int radius, OutputArray matrix, const int chn)
{
    int dimension = radius * 2 + 1;

    std::vector<Mat> channels;
    Mat oneChannel(dimension, dimension, CV_16SC1, Scalar(0));

    for (int i = 0; i < radius; i++)
    {
        oneChannel.col(i) = i - radius;
        oneChannel.col(dimension - 1 - i) = radius - i;
    }

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(oneChannel);
    }

    merge(channels, matrix);
}

void ft::FT12D_createPolynomMatrixHorizontal(int radius, OutputArray matrix, const int chn)
{
    int dimension = radius * 2 + 1;

    std::vector<Mat> channels;
    Mat oneChannel(dimension, dimension, CV_16SC1, Scalar(0));

    for (int i = 0; i < radius; i++)
    {
        oneChannel.row(i) = i - radius;
        oneChannel.row(dimension - 1 - i) = radius - i;
    }

    for (int i = 0; i < chn; i++)
    {
        channels.push_back(oneChannel);
    }

    merge(channels, matrix);
}

void ft::FT12D_inverseFT(cv::InputArray components, cv::InputArray kernel, cv::OutputArray output, int width, int height)
{
    Mat componentsMat = components.getMat();
    Mat kernelMat = kernel.getMat();

    CV_Assert(componentsMat.channels() == 1 && kernelMat.channels() == 1);

    int radiusX = (kernelMat.cols - 1) / 2;
    int radiusY = (kernelMat.rows - 1) / 2;
    int paddedOutputWidth = radiusX + width + kernelMat.cols;
    int paddedOutputHeight = radiusY + height + kernelMat.rows;

    output.create(height, width, CV_32F);

    Mat outputZeroes(paddedOutputHeight, paddedOutputWidth, CV_32F, Scalar(0));

    for (int i = 0; i < componentsMat.cols / kernelMat.cols; i++)
    {
        for (int o = 0; o < componentsMat.rows / kernelMat.rows; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernelMat.cols, kernelMat.rows);

            Mat component(componentsMat, Rect(i * kernelMat.cols, o * kernelMat.rows, kernelMat.cols, kernelMat.rows));

            Mat roiOutput(outputZeroes, area);
            roiOutput += kernelMat.mul(component);
        }
    }

    outputZeroes(Rect(radiusX, radiusY, width, height)).copyTo(output);
}

void ft::FT12D_inverseIrina(const Mat &c01, const Mat &c10, const Mat &kernel, Mat &S10, Mat &S01, Mat &iFT, int width, int height)
{
    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int paddedOutputWidth = radiusX + width + kernel.cols;
    int paddedOutputHeight = radiusY + height + kernel.rows;

    S10 = Mat::zeros(paddedOutputHeight, paddedOutputWidth, CV_32F);

    for (int i = 0; i < c10.cols; i++)
    {
        for (int o = 0; o < c10.rows; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiS10(S10, area);
            roiS10 += kernel.mul(c10.at<float>(o,i));
        }
    }

    S10 = S10(Rect(radiusX, radiusY, width, height));
}


void ft::FT12D_process(const cv::Mat &image, const cv::Mat &kernel, cv::Mat &output, const cv::Mat &mask)
{
    CV_Assert(image.channels() == kernel.channels());

    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int An = image.cols / radiusX + 1;
    int Bn = image.rows / radiusY + 1;
    int outputWidthPadded = radiusX + image.cols + kernel.cols;
    int outputHeightPadded = radiusY + image.rows + kernel.rows;

    Mat imagePadded;
    Mat maskPadded;

    output = Mat::zeros(outputHeightPadded, outputWidthPadded, CV_MAKETYPE(CV_32F, image.channels()));

    copyMakeBorder(image, imagePadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));

    Mat vecX;
    Mat vecY;

    ft::FT12D_createPolynomMatrixVertical(radiusX, vecX, image.channels());
    ft::FT12D_createPolynomMatrixHorizontal(radiusY, vecY, image.channels());

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiImage(imagePadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator00, numerator10, numerator01;
            multiply(roiImage, kernelMasked, numerator00, 1, CV_32F);
            multiply(numerator00, vecX, numerator10, 1, CV_32F);
            multiply(numerator00, vecY, numerator01, 1, CV_32F);

            Mat denominator00, denominator10, denominator01;
            denominator00 = kernelMasked;
            multiply(vecX.mul(vecX), kernelMasked, denominator10, 1, CV_32F);
            multiply(vecY.mul(vecY), kernelMasked, denominator01, 1, CV_32F);

            Scalar c00, c10, c01;
            divide(sum(numerator00), sum(denominator00), c00, 1, CV_32F);
            divide(sum(numerator10), sum(denominator10), c10, 1, CV_32F);
            divide(sum(numerator01), sum(denominator01), c01, 1, CV_32F);

            Mat component, updatedC10, updatedC01;

            multiply(c10, vecX, updatedC10, 1, CV_32F);
            multiply(c01, vecY, updatedC01, 1, CV_32F);

            add(updatedC01, updatedC10, component);
            add(component, c00, component);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(output, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    output = output(Rect(radiusX, radiusY, image.cols, image.rows));
}

void ft::DUMMY_ft1_inpaint(const cv::Mat &image, const cv::Mat &mask, cv::Mat &output, int radius)
{
    Mat kernel;
    ft::createKernel(ft::LINEAR, radius, kernel, image.channels());

    int radiusX = (kernel.cols - 1) / 2;
    int radiusY = (kernel.rows - 1) / 2;
    int An = image.cols / radiusX + 1;
    int Bn = image.rows / radiusY + 1;
    int outputWidthPadded = radiusX + image.cols + kernel.cols;
    int outputHeightPadded = radiusY + image.rows + kernel.rows;

    Mat imagePadded;
    Mat maskPadded;

    copyMakeBorder(image, imagePadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));
    copyMakeBorder(mask, maskPadded, radiusY, kernel.rows, radiusX, kernel.cols, BORDER_CONSTANT, Scalar(0));

    Mat vecX, vecY;

    ft::FT12D_createPolynomMatrixVertical(radiusX, vecX, image.channels());
    ft::FT12D_createPolynomMatrixHorizontal(radiusY, vecY, image.channels());

    Mat cMask = Mat::ones(Bn, An, CV_8U);
    output = Mat::zeros(outputHeightPadded, outputWidthPadded, CV_MAKETYPE(CV_32F, image.channels()));

    Mat c00(Bn, An, CV_32FC3);
    Mat c10(Bn, An, CV_32FC3);
    Mat c01(Bn, An, CV_32FC3);

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Mat roiImage(imagePadded, area);
            Mat roiMask(maskPadded, area);
            Mat kernelMasked;

            kernel.copyTo(kernelMasked, roiMask);

            Mat numerator00, numerator10, numerator01;
            multiply(roiImage, kernelMasked, numerator00, 1, CV_32F);
            multiply(numerator00, vecX, numerator10, 1, CV_32F);
            multiply(numerator00, vecY, numerator01, 1, CV_32F);

            Mat denominator00, denominator10, denominator01;
            denominator00 = kernelMasked;
            multiply(vecX.mul(vecX), kernelMasked, denominator10, 1, CV_32F);
            multiply(vecY.mul(vecY), kernelMasked, denominator01, 1, CV_32F);

            Scalar c00s, c10s, c01s;
            divide(sum(numerator00), sum(denominator00), c00s, 1, CV_32F);
            divide(sum(numerator10), sum(denominator10), c10s, 1, CV_32F);
            divide(sum(numerator01), sum(denominator01), c01s, 1, CV_32F);

            c00.row(o).col(i) = c00s;
            c10.row(o).col(i) = c10s;
            c01.row(o).col(i) = c01s;

            if (countNonZero(roiMask) < roiMask.rows * roiMask.cols)
            {
                cMask.row(o).col(i) = 0;
            }
        }
    }

    Mat c10Rec, c01Rec;

    ft::inpaint(c01, cMask, c01Rec, 2, ft::LINEAR, ft::ITERATIVE);
    ft::inpaint(c10, cMask, c10Rec, 2, ft::LINEAR, ft::ITERATIVE);

    for (int i = 0; i < An; i++)
    {
        for (int o = 0; o < Bn; o++)
        {
            int centerX = (i * radiusX) + radiusX;
            int centerY = (o * radiusY) + radiusY;
            Rect area(centerX - radiusX, centerY - radiusY, kernel.cols, kernel.rows);

            Scalar c00s, c01s, c10s;

            c00s = c00.at<Vec3f>(o,i);
            c10s = c10Rec.at<Vec3f>(o,i);
            c01s = c01Rec.at<Vec3f>(o,i);

            Mat component, updatedC10, updatedC01;

            multiply(c10s, vecX, updatedC10, 1, CV_32F);
            multiply(c01s, vecY, updatedC01, 1, CV_32F);

            add(updatedC01, updatedC10, component);
            add(component, c00s, component);

            Mat inverse;
            multiply(kernel, component, inverse, 1, CV_32F);

            Mat roiOutput(output, area);
            add(roiOutput, inverse, roiOutput);
        }
    }

    output = output(Rect(radiusX, radiusY, image.cols, image.rows));
}
