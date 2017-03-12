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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
#include "npy_blob.hpp"

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

template<typename TString>
static std::string _tf(TString filename)
{
    return (getOpenCVExtraDir() + "/dnn/") + filename;
}

TEST(Test_Caffe, read_gtsrb)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("gtsrb.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

TEST(Test_Caffe, read_googlenet)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("bvlc_googlenet.prototxt"), "");
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }
}

#if defined(ENABLE_CAFFE_MODEL_TESTS)

#if defined(ENABLE_CAFFE_ALEXNET_TEST) //AlexNet is disabled now

TEST(Reproducibility_AlexNet, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("bvlc_alexnet.prototxt"), _tf("bvlc_alexnet.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Mat sample = imread(_tf("grace_hopper_227.png"));
    ASSERT_TRUE(!sample.empty());
    cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);

    Size inputSize(227, 227);

    if (sample.size() != inputSize)
        resize(sample, sample, inputSize);

    net.setBlob(".data", dnn::Blob::fromImages(sample));
    net.forward();

    Blob out = net.getBlob("prob");
    Blob ref = blobFromNPY(_tf("caffe_alexnet_prob.npy"));
    normAssert(ref, out);
}

#endif

#if defined(ENABLE_CAFFE_FCN_TEST)

TEST(Reproducibility_FCN, Accuracy)
{
    Net net;
    {
        Ptr<Importer> importer = createCaffeImporter(_tf("fcn8s-heavy-pascal.prototxt"), _tf("fcn8s-heavy-pascal.caffemodel"));
        ASSERT_TRUE(importer != NULL);
        importer->populateNet(net);
    }

    Mat sample = imread(_tf("street.png"));
    ASSERT_TRUE(!sample.empty());

    Size inputSize(500, 500);
    if (sample.size() != inputSize)
        resize(sample, sample, inputSize);

    cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);

    net.setBlob(".data", dnn::Blob::fromImages(sample));
    net.forward();

    Blob out = net.getBlob("score");

    Blob ref = blobFromNPY(_tf("caffe_fcn8s_prob.npy"));
    normAssert(ref, out);
}

#endif

#endif

}
