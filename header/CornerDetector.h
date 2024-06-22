#pragma once
#ifndef CORNERDETECTOR_H
#define CORNERDETECTOR_H

#include "lib.h"

class CornerDetector {
private:
public:
    int detect_corner_Harris(const cv::Mat& source_img, cv::Mat& dest_img, int block_size, int aperture_size, int k);

    CornerDetector();
    ~CornerDetector();
};

#endif