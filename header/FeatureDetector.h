#pragma once
#ifndef FEATUREDETECTOR_H
#define FEATUREDETECTOR_H

#include "lib.h"
#include <opencv2/features2d.hpp>

class SIFTDetector
{
private:
    std::vector<std::vector<cv::Mat>> buildGaussianPyramid(const cv::Mat &source_img, int num_octaves, int num_scales);
    std::vector<std::vector<cv::Mat>> buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &gauss_pyramid);

    void detectExtrema(const std::vector<std::vector<cv::Mat>>& dog_pyramid, std::vector<cv::KeyPoint>& keypoints, float contrast_threshold);
    void assignOrientations(const std::vector<std::vector<cv::Mat>> &gauss_pyramid, std::vector<cv::KeyPoint> &keypoints);

    cv::Mat generateDescriptor(const cv::Mat &source_img, const cv::KeyPoint &keypoint);

public:
    SIFTDetector();
    ~SIFTDetector();

    std::vector<cv::KeyPoint> detectKeypoints(const cv::Mat &source_img, float contrast_threshold);
    cv::Mat computeDescriptors(const cv::Mat &source_img, std::vector<cv::KeyPoint> &keypoints);

    void drawKeypoints(const cv::Mat& source_img, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& dest_img);
};

#endif