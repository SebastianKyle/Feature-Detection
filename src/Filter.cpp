#include "../header/Filter.h"

Filter::Filter() {

}

Filter::~Filter() {

}

int Filter::avg_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k) {
    if (!source_img.data) return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);
            int convolve = 0;

            // Border: pad with reflection
            for (int yi = y - quarter_side * (y - quarter_side >= 0); yi <= y + quarter_side; yi++) {
                for (int xi = x - quarter_side * (x - quarter_side >= 0); xi <= x + quarter_side; xi++) {
                    convolve += source_img.at<uchar>(yi, xi);
                }
            }

            convolve = round(convolve / (k * k));

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve);
        }     
    }

    dest_img = output.clone();
    return 1;
}

int Filter::median_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k) {
    if (!source_img.data) return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);

            int median = -1;
            int count_greater = -1;
            bool median_found = false;

            // Border: pad with reflection
            for (int yi = y - quarter_side * (y - quarter_side >= 0); yi <= y + quarter_side && !median_found; yi++) {
                for (int xi = x - quarter_side * (x - quarter_side >= 0); xi <= x + quarter_side && !median_found; xi++) {
                    if (source_img.at<uchar>(yi, xi) >= median) {
                        median = source_img.at<uchar>(yi, xi);
                        count_greater++;

                        if (count_greater >= (k * k - 1) / 2) {
                            median_found = true;
                        }
                    }
                }
            }

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(median);
        }     
    }

    dest_img = output.clone();
    return 1;
}

int Filter::gaussian_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k) { 
    if (!source_img.data) return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);
            
            int convolve = 0;
            double sigma = k / (2 * 3.14);

            // Border: pad with reflection
            for (int yi = y - quarter_side * (y - quarter_side >= 0); yi <= y + quarter_side; yi++) {
                for (int xi = x - quarter_side * (x - quarter_side >= 0); xi <= x + quarter_side; xi++) {
                    convolve += source_img.at<uchar>(yi, xi) * 
                                    round((1/(2 * 3.14 * sigma * sigma)) 
                                    * exp(-((abs(y - yi) + 1) * (abs(y - yi) + 1) + (abs(x - xi) + 1) * (abs(x - xi) + 1)) / (2 * sigma * sigma)));
                }
            }

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve);
        }     
    }

    dest_img = output.clone();
    return 1;
}

int Filter::bilateral_filter(const cv::Mat& source_img, cv::Mat& dest_img, int k) {
    if (!source_img.data) return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);
            
            int convolve = 0;
            double sigma = k / (2 * 3.14);
            double wsb = 0;

            // Border: pad with reflection
            for (int yi = y - quarter_side * (y - quarter_side >= 0); yi <= y + quarter_side; yi++) {
                for (int xi = x - quarter_side * (x - quarter_side >= 0); xi <= x + quarter_side; xi++) {
                    // Spatial gaussian
                    double n_sigma_s = (1/(2 * 3.14 * sigma * sigma)) 
                        * exp(-((abs(y - yi) + 1) * (abs(y - yi) + 1) + (abs(x - xi) + 1) * (abs(x - xi) + 1)) / (2 * sigma * sigma));

                    // Brightness gaussian
                    double n_sigma_b = (1/(2 * 3.14 * sigma * sigma))
                        * exp(-((source_img.at<uchar>(yi, xi) - source_img.at<uchar>(y, x)) * (source_img.at<uchar>(yi, xi) - source_img.at<uchar>(y, x))) / (2 * sigma * sigma));

                    convolve += source_img.at<uchar>(yi, xi) * n_sigma_s * n_sigma_b;
                    wsb += n_sigma_s * n_sigma_b;
                }
            }
            convolve = round(convolve / wsb);

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve);
        }     
    }

    dest_img = output.clone();
    return 1;
}