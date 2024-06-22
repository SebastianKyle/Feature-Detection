#include "CornerDetector.h"

CornerDetector::CornerDetector() {}

CornerDetector::~CornerDetector() {}

int CornerDetector::detect_corner_Harris(const cv::Mat &source_img, cv::Mat &dest_img, int block_size, int aperture_size, int k)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat gray_img;
    cv::cvtColor(source_img, gray_img, cv::COLOR_RGB2GRAY);
    gray_img.convertTo(gray_img, CV_32F);

    // Gradients
    cv::Mat Ix, Iy;
    cv::Sobel(gray_img, Ix, CV_32F, 1, 0, aperture_size);
    cv::Sobel(gray_img, Iy, CV_32F, 0, 1, aperture_size);

    // Second moments
    cv::Mat Ix2 = Ix.mul(Ix);
    cv::Mat Iy2 = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // Second moments for pixels neighbor
    cv::Mat Sx2, Sy2, Sxy;
    cv::boxFilter(Ix2, Sx2, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(Iy2, Sy2, CV_32F, cv::Size(block_size, block_size));
    cv::boxFilter(Ixy, Sxy, CV_32F, cv::Size(block_size, block_size));

    cv::Mat response = cv::Mat::zeros(gray_img.size(), CV_32F);
    for (int y = 0; y < gray_img.rows; y++)
    {
        for (int x = 0; x < gray_img.cols; x++)
        {
            float detM = (Sx2.at<float>(y, x) * Sy2.at<float>(y, x)) - (Sxy.at<float>(y, x) * Sxy.at<float>(y, x));
            float traceM = Sx2.at<float>(y, x) + Sy2.at<float>(y, x);
            response.at<float>(y, x) = detM - k * traceM * traceM;
        }
    }

    cv::Mat response_norm;
    cv::normalize(response, response_norm, 0, 255, cv::NORM_MINMAX, CV_32F, cv::Mat());
    response_norm.convertTo(response_norm, CV_8U);

    // Non-maximal suppression
    int suppression_window = 5;
    cv::Mat suppressed_response = cv::Mat::zeros(response.size(), CV_8U);

    for (int y = suppression_window; y < response_norm.rows - suppression_window; y++)
    {
        for (int x = suppression_window; x < response_norm.cols - suppression_window; x++)
        {
            float center_value = response_norm.at<uchar>(y, x);
            bool is_max = true;

            for (int wy = -suppression_window; wy <= suppression_window; wy++)
            {
                for (int wx = -suppression_window; wx <= suppression_window; wx++)
                {
                    if (response_norm.at<uchar>(y + wy, x + wx) > center_value)
                    {
                        is_max = false;
                        break;
                    }
                }

                if (!is_max)
                    break;
            }

            if (is_max && center_value > 100)
            {
                suppressed_response.at<uchar>(y, x) = center_value;
            }
        }
    }

    dest_img = source_img.clone();
    for (int y = 0; y < suppressed_response.rows; y++)
    {
        for (int x = 0; x < suppressed_response.cols; x++)
        {
            if (suppressed_response.at<uchar>(y, x) > 0)
            {
                cv::circle(dest_img, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), 2);
            }
        }
    }

    return 1;
}