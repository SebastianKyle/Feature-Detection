#include "EdgeDetector.h"

EdgeDetector::EdgeDetector()
{
}

EdgeDetector::~EdgeDetector()
{
}

/* ---------- Kernel ---------- */
int EdgeDetector::x_gradient_sobel(cv::Mat image, int x, int y)
{
    /*
        1 0 -1
        2 0 -2
        1 0 -1
    */

    return image.at<uchar>(y - 1, x - 1) + 2 * image.at<uchar>(y, x - 1) + image.at<uchar>(y + 1, x - 1) - image.at<uchar>(y - 1, x + 1) - 2 * image.at<uchar>(y, x + 1) - image.at<uchar>(y + 1, x + 1);
}

int EdgeDetector::y_gradient_sobel(cv::Mat image, int x, int y)
{
    /*
         1  2  1
         0  0  0
        -1 -2 -1
    */

    return image.at<uchar>(y - 1, x - 1) + 2 * image.at<uchar>(y - 1, x) + image.at<uchar>(y - 1, x + 1) - image.at<uchar>(y + 1, x - 1) - 2 * image.at<uchar>(y + 1, x) - image.at<uchar>(y + 1, x + 1);
}

/* ---------- Detect ---------- */

int EdgeDetector::detect_by_sobel(const cv::Mat &source_img, cv::Mat &dest_img)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();

    int gx = 0, gy = 0;
    int grad_magnitude = 0;

    int width = source_img.cols;
    int height = source_img.rows;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            gx = x_gradient_sobel(source_img, x, y);
            gy = y_gradient_sobel(source_img, x, y);

            grad_magnitude = floor(sqrt(gx * gx + gy * gy));
            grad_magnitude = grad_magnitude > 110 ? grad_magnitude : 0;

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(grad_magnitude);
        }
    }

    dest_img = output.clone();
    return 1;
}

int EdgeDetector::detect_by_laplace(const cv::Mat &source_img, cv::Mat &dest_img, float threshold)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = cv::Mat::zeros(source_img.size(), CV_8U);
    int width = source_img.cols, height = source_img.rows;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int convolve = -source_img.at<uchar>(y - 1, x) - source_img.at<uchar>(y, x - 1) + 4 * source_img.at<uchar>(y, x) - source_img.at<uchar>(y, x + 1) - source_img.at<uchar>(y + 1, x);

            bool zero_crossing = false;

            float neighbors[8] = {
                source_img.at<uchar>(y - 1, x - 1),
                source_img.at<uchar>(y - 1, x),
                source_img.at<uchar>(y - 1, x + 1),
                source_img.at<uchar>(y, x - 1),
                source_img.at<uchar>(y, x + 1),
                source_img.at<uchar>(y + 1, x - 1),
                source_img.at<uchar>(y + 1, x),
                source_img.at<uchar>(y + 1, x + 1)};

            for (int i = 0; i < 8; i++)
            {
                if ((convolve > threshold && neighbors[i] < -threshold) ||
                    (convolve < -threshold && neighbors[i] > threshold))
                {
                    zero_crossing = true;
                    break;
                }
            }

            if (zero_crossing)
            {
                output.at<uchar>(y, x) = 255;
            }
        }
    }

    dest_img = output.clone();
    return 1;
}

int EdgeDetector::detect_by_canny(const cv::Mat &source_img, cv::Mat &dest_img, int min_grad, int max_grad)
{
    if (!source_img.data)
        return 0;

    if (max_grad < 0)
        max_grad = 0;
    if (min_grad < 0)
        min_grad = 0;
    if (min_grad > max_grad)
        min_grad = max_grad;

    cv::Mat gray_img;
    cv::cvtColor(source_img, gray_img, cv::COLOR_BGR2GRAY);

    int width = gray_img.cols, height = gray_img.rows;

    double gx = 0, gy = 0;
    std::vector<double> row_grads;
    std::vector<std::vector<double>> grads(height, std::vector<double>(width));
    std::vector<int> row_angles;
    std::vector<std::vector<int>> angles(height, std::vector<int>(width));

    int Gmax = 0;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            gx = x_gradient_sobel(gray_img, x, y);
            gy = y_gradient_sobel(gray_img, x, y);

            double edge_grad = sqrt(gx * gx + gy * gy);

            if (edge_grad > Gmax)
            {
                Gmax = edge_grad;
            }

            grads[y][x] = edge_grad;

            double angle = atan2(gy, gx) * 100 / CV_PI;

            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180) || (angle >= -22.5 && angle < 0) ||
                (angle >= -180 && angle < -157.5))
                angles[y][x] = 0;
            else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5))
                angles[y][x] = 45;
            else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5))
                angles[y][x] = 90;
            else if ((angle >= 112.5 && angle < 157.5) || (angle >= -67.5 && angle < -22.5))
                angles[y][x] = 135;
        }
    }

    cv::Mat nonmax_suppressed = cv::Mat::zeros(height, width, CV_8UC1);

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int angle = angles[y][x];
            double magnitude = grads[y][x];

            bool is_edge = true;

            // If a neighboring pixel in the gradient direction has a larger gradient magnitude, it's not an edge
            if (angle == 0)
            {
                if (grads[y][x - 1] > magnitude || grads[y][x + 1] > magnitude)
                    is_edge = false;
            }
            else if (angle == 45)
            {
                if (grads[y - 1][x + 1] > magnitude || grads[y + 1][x - 1] > magnitude)
                    is_edge = false;
            }
            else if (angle == 90)
            {
                if (grads[y - 1][x] > magnitude || grads[y + 1][x] > magnitude)
                    is_edge = false;
            }
            else if (angle == 135)
            {
                if (grads[y - 1][x - 1] > magnitude || grads[y + 1][x + 1] > magnitude)
                    is_edge = false;
            }

            if (is_edge)
            {
                if (magnitude > max_grad) // strong edge
                    nonmax_suppressed.at<uchar>(y, x) = 255;
                else if (magnitude >= min_grad) // not sure (save for later)
                    nonmax_suppressed.at<uchar>(y, x) = 128;
                else // weak edge
                    nonmax_suppressed.at<uchar>(y, x) = 0;
            }
        }
    }

    // Hysteresis thresholding for "not sure" edges
    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            if (nonmax_suppressed.at<uchar>(y, x) == 128)
            {
                bool is_connected_to_strong = false;

                // Check 8-neighbors
                for (int ky = -1; ky <= 1; ky++)
                {
                    for (int kx = -1; kx <= 1; kx++)
                    {
                        if (nonmax_suppressed.at<uchar>(y + ky, x + kx) == 255)
                        {
                            is_connected_to_strong = true;
                            break;
                        }
                    }
                    if (is_connected_to_strong)
                        break;
                }

                if (is_connected_to_strong)
                    nonmax_suppressed.at<uchar>(y, x) = 255;
                else
                    nonmax_suppressed.at<uchar>(y, x) = 0;
            }
        }
    }

    dest_img = nonmax_suppressed.clone();
    return 1;
}