#pragma once
#ifndef EDGEDETECTOR_H
#define EDGEDETECTOR_H

#include "lib.h"

class EdgeDetector {
private:

public:
    int x_gradient_sobel(cv::Mat img, int x, int y);
    int y_gradient_sobel(cv::Mat img, int x, int y);

    int detect_by_sobel(const cv::Mat& source_img, cv::Mat& dest_img);
    int detect_by_laplace(const cv::Mat& source_img, cv::Mat& dest_img);
    int detect_by_canny(const cv::Mat& source_img, cv::Mat& dest_img);

    EdgeDetector();
    ~EdgeDetector();
};

#endif