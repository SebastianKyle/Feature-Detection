#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "lib.h"

class Filter {
private: 

public: 
    int rgb2gray_filter(const cv::Mat& source_img, cv::Mat& dest_img);
    int avg_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);
    int median_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);
    int gaussian_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);
    int bilateral_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k, int sigma_b);

    Filter();
    ~Filter();
};

#endif