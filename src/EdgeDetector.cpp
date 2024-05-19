#include "../header/EdgeDetector.h"

EdgeDetector::EdgeDetector() {

}

EdgeDetector::~EdgeDetector() {

}

/* ---------- Kernel ---------- */
int EdgeDetector::x_gradient_sobel(cv::Mat image, int x, int y) {
    /*
        1 0 -1
        2 0 -2
        1 0 -1
    */

    return image.at<uchar>(y - 1, x - 1) + 2 * image.at<uchar>(y, x - 1) + image.at<uchar>(y + 1, x - 1)
            - image.at<uchar>(y - 1, x + 1) - 2 * image.at<uchar>(y, x + 1) - image.at<uchar>(y + 1, x + 1);
}

int EdgeDetector::y_gradient_sobel(cv::Mat image, int x, int y) {
    /*
         1  2  1
         0  0  0
        -1 -2 -1
    */

    return image.at<uchar>(y - 1, x - 1) + 2 * image.at<uchar>(y - 1, x) + image.at<uchar>(y - 1, x + 1)
            - image.at<uchar>(y + 1, x - 1) - 2 * image.at<uchar>(y + 1, x) - image.at<uchar>(y + 1, x + 1);
}

/* ---------- Detect ---------- */

int EdgeDetector::detect_by_sobel(const cv::Mat& source_img, cv::Mat& dest_img) {
    if (!source_img.data)
        return 0;
    
    cv::Mat output = source_img.clone();

    int gx = 0, gy = 0;
    int grad_magnitude = 0;

    int width = source_img.cols;
    int height = source_img.rows;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            gx = x_gradient_sobel(source_img, x, y);
            gy = y_gradient_sobel(source_img, x, y);

            grad_magnitude = floor(sqrt(gx * gx + gy * gy));

            grad_magnitude = grad_magnitude > 255 ? 255 : grad_magnitude;
            grad_magnitude = grad_magnitude < 0 ? 0 : grad_magnitude;

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(grad_magnitude);
        }
    }

    dest_img = output.clone();
    return 1;
}

int EdgeDetector::detect_by_laplace(const cv::Mat& source_img, cv::Mat& dest_img) {
    if (!source_img.data) return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            double epsilon = 0.5;

            int convolve = source_img.at<uchar>(y - 1, x - 1) + 4 * source_img.at<uchar>(y - 1, x) + source_img.at<uchar>(y - 1, x + 1)
                        + 4 * source_img.at<uchar>(y, x - 1) - 20 * source_img.at<uchar>(y, x) + source_img.at<uchar>(y, x + 1)
                        + source_img.at<uchar>(y + 1, x - 1) + 4 * source_img.at<uchar>(y + 1, x) + source_img.at<uchar>(y + 1, x + 1);
            int laplacian = floor((1/(6 * epsilon * epsilon)) * convolve);
            
            laplacian = laplacian > 255 ? 255 : laplacian;
            laplacian = laplacian < 0 ? 0 : laplacian;

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(laplacian);
        }
    }

    dest_img = output.clone();
    return 1;
}

int EdgeDetector::detect_by_canny(const cv::Mat& source_img, cv::Mat& dest_img) {
    // TODO: implement canny
    return 1;
}