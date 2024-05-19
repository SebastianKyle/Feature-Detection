#pragma once
#ifndef FILTER_H
#define FILTER_H

#include "lib.h"

class Filter {
private: 

public: 
    int avg_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);
    int median_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);
    int gaussian_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);
    int bilateral_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k);

    Filter();
    ~Filter();
};

#endif