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

int EdgeDetector::detect_by_laplace(const cv::Mat &source_img, cv::Mat &dest_img)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            double epsilon = 0.7;

            int convolve = source_img.at<uchar>(y - 1, x - 1) + 4 * source_img.at<uchar>(y - 1, x) + source_img.at<uchar>(y - 1, x + 1) + 4 * source_img.at<uchar>(y, x - 1) - 20 * source_img.at<uchar>(y, x) + 4 * source_img.at<uchar>(y, x + 1) + source_img.at<uchar>(y + 1, x - 1) + 4 * source_img.at<uchar>(y + 1, x) + source_img.at<uchar>(y + 1, x + 1);

            // "Zero-crossing"
            int laplacian = int((1 / (6 * epsilon * epsilon)) * convolve + 128);
            laplacian = abs(laplacian - 128) <= 80 ? 0 : 255;

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(laplacian);
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

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    double gx = 0, gy = 0;
    std::vector<double> row_grads;
    std::vector<std::vector<double>> grads;
    std::vector<int> row_angles;
    std::vector<std::vector<int>> angles;

    int Gmax = 0;

    for (int y = 1; y < height - 1; y++)
    {
        row_grads.clear();
        row_angles.clear();

        for (int x = 1; x < width - 1; x++)
        {
            gx = x_gradient_sobel(source_img, x, y);
            gy = y_gradient_sobel(source_img, x, y);

            // Gradient magnitude
            double edge_grad = sqrt(gx * gx + gy * gy);

            if (edge_grad > Gmax)
                Gmax = edge_grad;

            row_grads.push_back(edge_grad);

            // Gradient direction
            double angle = atan2(gy, gx) * 180 / 3.14;

            // Map the angle to discrete values
            if ((angle >= 0 && angle < 22.5) || (angle > 157.5 && angle < 180) || (angle > -22.5 && angle < 0) ||
                (angle > -180 && angle < -157.5))
                angle = 0;
            else if ((angle > 22.5 && angle < 67.5) || (angle > -157.5 && angle < -112.5))
                angle = 45;
            else if ((angle >= 67.5 && angle <= 112.5) || (angle > -112.5 && angle < -67.5))
                angle = 90;
            else if ((angle > 112.5 && angle <= 157.5) || (angle > -67.5 && angle < -22.5))
                angle = 135;

            row_angles.push_back(angle);
        }

        grads.push_back(row_grads);
        angles.push_back(row_angles);
    }

    // if (height > Gmax)
    //     height = Gmax;

    // Non-max suppression
    for (int y = 1; y < height - 3; y++)
    {
        for (int x = 1; x < width - 3; x++)
        {
            int value = grads[y][x];

            // If a neighboring pixel to a direction has larger grad magnitude then
            // the current pixel, it's not an edge, label it as 0
            if (angles[y][x] == 45)
            {
                if (grads[y][x] < std::max(grads[y - 1][x - 1], grads[y + 1][x + 1]))
                    value = 0;
            }
            else if (angles[y][x] == 90)
            {
                if (grads[y][x] < std::max(grads[y - 1][x], grads[y + 1][x]))
                    value = 0;
            }
            else if (angles[y][x] == 135)
            {
                if (grads[y][x] < std::max(grads[y - 1][x + 1], grads[y + 1][x - 1]))
                    value = 0;
                else if (grads[y][x] < std::max(grads[y][x - 1], grads[y][x + 1]))
                    value = 0;
            }

            // Classify strong, weak and non-edge
            if (value > max_grad)
                value = 255;
            else if (value < min_grad)
                value = 0;
            else
                value = 128;

            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(value);
        }
    }

    int non_max_height = output.rows, non_max_width = output.cols;

    // Hysteresis thresholding
    // Is a weak edge in fact a strong edge ?
    for (int y = 1; y < non_max_height - 2; y++)
    {
        for (int x = 1; x < non_max_width - 2; x++)
        {
            if (output.at<uchar>(y, x) == 128)
            {
                // If there's a neighbor of the pixel is a strong edge
                // => it's a strong edge
                if ((output.at<uchar>(y - 1, x - 1) == 255) || (output.at<uchar>(y - 1, x) == 255) || (output.at<uchar>(y - 1, x + 1) == 255) ||
                    (output.at<uchar>(y, x - 1) == 255) || (output.at<uchar>(y, x + 1) == 255) ||
                    (output.at<uchar>(y + 1, x - 1) == 255) || (output.at<uchar>(y + 1, x) == 255) || (output.at<uchar>(y + 1, x + 1) == 255))
                    output.at<uchar>(y, x) = cv::saturate_cast<uchar>(255);
                else
                    output.at<uchar>(y, x) = cv::saturate_cast<uchar>(0);
            }
        }
    }

    dest_img = output.clone();

    return 1;
}