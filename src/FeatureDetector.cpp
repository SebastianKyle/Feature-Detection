#include "FeatureDetector.h"

SIFTDetector::SIFTDetector() {}

SIFTDetector::~SIFTDetector() {}

/************************************************************************************************
 * Buid Gaussian pyramid of octaves contains blurred images.
 */
std::vector<std::vector<cv::Mat>> SIFTDetector::buildGaussianPyramid(const cv::Mat &source_img, int num_octaves, int num_scales)
{
    std::vector<std::vector<cv::Mat>> gauss_pyramid(num_octaves);

    // double k = pow(2, 1 / (num_scales));
    // double initial_sigma = 1.6;
    for (int i = 0; i < num_octaves; ++i)
    {
        gauss_pyramid[i].resize(num_scales + 3);

        // double sigma = initial_sigma * pow(2, i);
        for (int j = 0; j < num_scales + 3; ++j)
        {
            if (i == 0 && j == 0)
            {
                gauss_pyramid[i][j] = source_img.clone();
            }
            else if (j == 0)
            {
                cv::resize(gauss_pyramid[i - 1][num_scales], gauss_pyramid[i][j], cv::Size(gauss_pyramid[i - 1][num_scales].cols / 2, gauss_pyramid[i - 1][num_scales].rows / 2));
            }
            else
            {
                // sigma *= k;
                double sigma = pow(2, j / static_cast<double>(num_scales));
                cv::GaussianBlur(gauss_pyramid[i][j - 1], gauss_pyramid[i][j], cv::Size(0, 0), sigma);
            }
        }
    }
    return gauss_pyramid;
}

/************************************************************************************************
 * Build Difference of Gaussian (DoG) images pyramid by subtracting subsequent gaussian blurred
 * image pairs in the Gaussian pyramid.
 */
std::vector<std::vector<cv::Mat>> SIFTDetector::buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &gauss_pyramid)
{
    int num_octaves = gauss_pyramid.size();
    int num_scales = gauss_pyramid[0].size() - 1;
    std::vector<std::vector<cv::Mat>> dog_pyramid(num_octaves);

    for (int i = 0; i < num_octaves; ++i)
    {
        dog_pyramid[i].resize(num_scales);
        for (int j = 0; j < num_scales; ++j)
        {
            dog_pyramid[i][j] = gauss_pyramid[i][j + 1] - gauss_pyramid[i][j];
        }
    }

    return dog_pyramid;
}

/************************************************************************************************
 * Obtain key points by detecting extremas in 3x3 regions of current and adjacent scale images.
 *
 * The center pixel is considered an extrema if it is greater or less than all 26 neighbor pixels
 * in the 3x3 region.
 *
 * The keypoint is further refined to obtain sub-pixel accuracy.
 */
void SIFTDetector::detectExtrema(const std::vector<std::vector<cv::Mat>> &dog_pyramid, std::vector<cv::KeyPoint> &keypoints, float contrast_threshold)
{
    for (int octave = 0; octave < dog_pyramid.size(); octave++)
    {
        const auto& dog_images = dog_pyramid[octave];
        float k = std::pow(2, 1.0 / dog_images.size());

        for (int scale = 1; scale < dog_images.size() - 1; scale++)
        {
            const cv::Mat &prev = dog_images[scale - 1];
            const cv::Mat &curr = dog_images[scale];
            const cv::Mat &next = dog_images[scale + 1];

            int width = curr.cols, height = curr.rows;

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    float val = curr.at<float>(y, x);

                    if (std::abs(val) >= contrast_threshold)
                    {
                        bool is_maxima = true;
                        bool is_minima = true;
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                for (int dz = -1; dz <= 1; dz++)
                                {
                                    if (dy == 0 && dx == 0 && dz == 0)
                                        continue;

                                    float neighbor_val = 0;
                                    if (dz == -1)
                                    {
                                        neighbor_val = prev.at<float>(y + dy, x + dx);
                                    }
                                    else if (dz == 0)
                                    {
                                        neighbor_val = curr.at<float>(y + dy, x + dx);
                                    }
                                    else if (dz == 1)
                                    {
                                        neighbor_val = next.at<float>(y + dy, x + dx);
                                    }

                                    if (val <= neighbor_val)
                                        is_maxima = false;
                                    if (val >= neighbor_val)
                                        is_minima = false;

                                    if (!is_maxima && !is_minima)
                                        break;
                                }
                                if (!is_maxima && !is_minima)
                                    break;
                            }
                            if (!is_maxima && !is_minima)
                                break;
                        }

                        // Subpixel refinement
                        if (is_minima || is_maxima)
                        {
                            cv::KeyPoint kp;
                            // kp.pt.x = x;
                            // kp.pt.y = y;
                            kp.pt = cv::Point2f(x * pow(2, octave), y * pow(2, octave));
                            // kp.size = pow(2, octave + scale / static_cast<float>(dog_images.size()));
                            // kp.size = 1.6 * pow(2, (scale + octave) / static_cast<float>(dog_images.size()));
                            // kp.size = 1.6 * pow(2, (scale + octave));
                            // kp.size = pow(2, scale / static_cast<double>(dog_images.size()));
                            // float sigma = 1.6 * std::pow(k, scale);
                            // kp.size = sigma * pow(2, octave) * 5;
                            kp.size = 1.6 * pow(2, octave) * pow(2, scale / dog_images.size());

                            // Gradient of D(x, y, sigma)
                            cv::Matx31f dD(
                                (curr.at<float>(y, x + 1) - curr.at<float>(y, x - 1)) / 2.0f,
                                (curr.at<float>(y + 1, x) - curr.at<float>(y - 1, x)) / 2.0f,
                                (next.at<float>(y, x) - prev.at<float>(y, x)) / 2.0f);

                            // Hessian matrix of D(x, y, sigma)
                            cv::Matx33f H(
                                (curr.at<float>(y, x + 1) + curr.at<float>(y, x - 1) - 2 * val),
                                (curr.at<float>(y + 1, x) - curr.at<float>(y - 1, x + 1)) / 4.0f,
                                (next.at<float>(y, x) - prev.at<float>(y, x + 1)) / 4.0f,

                                (prev.at<float>(y - 1, x) - prev.at<float>(y - 1, x + 1)) / 4.0f,
                                (curr.at<float>(y + 1, x) + curr.at<float>(y - 1, x) - 2 * val),
                                (next.at<float>(y, x - 1) - prev.at<float>(y, x + 1)) / 4.0f,

                                (next.at<float>(y, x + 1) - prev.at<float>(y, x + 1)) / 4.0f,
                                (next.at<float>(y + 1, x) - prev.at<float>(y + 1, x)) / 4.0f,
                                (next.at<float>(y, x) + prev.at<float>(y, x) - 2 * val));

                            cv::Matx31f offset = -H.inv() * dD;

                            kp.pt.x += offset(0);
                            kp.pt.y += offset(1);

                            if (std::abs(offset(0)) < 0.5f && std::abs(offset(1)) < 0.5f)
                            {
                                keypoints.push_back(kp);
                            }
                        }
                    }
                }
            }
        }
    }
}

/************************************************************************************************
 * Assign orientation to keypoints using y and x-direction gradients
 */
void SIFTDetector::assignOrientations(const std::vector<std::vector<cv::Mat>> &gauss_pyramid, std::vector<cv::KeyPoint> &keypoints)
{
    for (auto &kp : keypoints)
    {
        int x = kp.pt.x;
        int y = kp.pt.y;

        cv::Mat grad_x, grad_y;
        cv::Sobel(gauss_pyramid[0][0], grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gauss_pyramid[0][0], grad_y, CV_32F, 0, 1, 3);

        float angle = atan2(grad_y.at<float>(y, x), grad_x.at<float>(y, x)) * 180 / CV_PI;
        kp.angle = angle;
    }
}

/************************************************************************************************
 * Generate SIFT descriptor for keypoints
 *
 * Use a region of size 16x16 containing 16 sub-regions
 * Each sub-regions is of size 4x4
 *
 * Construct orientation histogram with respect to keypoint angle for each sub-regions
 * Concatenate and normalize histograms to get keypoint descriptor
 */
cv::Mat SIFTDetector::generateDescriptor(const cv::Mat &source_img, const cv::KeyPoint &keypoint)
{
    const int num_bins = 8;         // number of bins in orientation histogram
    const int window_size = 4;      // size of each region window
    const float scale_factor = 1.5; // scale factor for gaussian window
    const int region_size = 4;      // amount of regions with respect to one axis

    // Gaussian window
    cv::Mat gauss_window = cv::getGaussianKernel(window_size * region_size, scale_factor * keypoint.size, CV_32F);
    gauss_window = gauss_window * gauss_window.t();

    // Descriptor
    cv::Mat descriptor = cv::Mat::zeros(1, region_size * region_size * num_bins, CV_32F);

    int half_window = window_size * region_size / 2;
    float angle_grad = keypoint.angle * CV_PI / 180.0;

    float cos_theta = cos(angle_grad);
    float sin_theta = sin(angle_grad);

    for (int dy = -half_window; dy < half_window; dy++)
    {
        for (int dx = -half_window; dx < half_window; dx++)
        {
            int rotated_x = cos_theta * dx - sin_theta * dy;
            int rotated_y = sin_theta * dx + cos_theta * dy;

            int img_x = keypoint.pt.x + rotated_x;
            int img_y = keypoint.pt.y + rotated_y;

            if (img_x < 0 || img_x >= source_img.cols - 1 ||
                img_y < 0 || img_y >= source_img.rows - 1)
            {
                continue;
            }

            float grad_x = source_img.at<float>(img_y, img_x + 1) - source_img.at<float>(img_y, img_x - 1);
            float grad_y = source_img.at<float>(img_y + 1, img_x) - source_img.at<float>(img_y - 1, img_x);
            float magnitude = sqrt(grad_x * grad_x + grad_y * grad_y);
            float orientation = atan2(grad_y, grad_x) * 180 / CV_PI;

            float weight = gauss_window.at<float>(dy + half_window, dx + half_window);

            // Map orientation to [0, 360)
            orientation = fmod(orientation - keypoint.angle + 360.0, 360.0);

            int bin = round(orientation * num_bins / 360.0);
            if (bin >= num_bins)
                bin -= num_bins;

            int region_x = (dx + half_window) / region_size;
            int region_y = (dy + half_window) / region_size;

            int descriptor_index = (region_y * region_size + region_x) * num_bins + bin;

            descriptor.at<float>(0, descriptor_index) += weight * magnitude;
        }
    }

    cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);

    return cv::Mat();
}

/************************************************************************************************
 * Detect keypoints from DoG pyramid and assign orientations.
 */
std::vector<cv::KeyPoint> SIFTDetector::detectKeypoints(const cv::Mat &source_img, float contrast_threshold)
{
    std::vector<cv::KeyPoint> keypoints;
    int num_octaves = 4;
    int num_scales = 5;

    cv::Mat input_img = source_img.clone();
    if (input_img.channels() == 3) {
        cv::cvtColor(input_img, input_img, cv::COLOR_BGR2GRAY);
    }
    input_img.convertTo(input_img, CV_32F, 1.0 / 255.0);

    std::vector<std::vector<cv::Mat>> gauss_pyramid = buildGaussianPyramid(input_img, num_octaves, num_scales);
    std::vector<std::vector<cv::Mat>> dog_pyramid = buildDoGPyramid(gauss_pyramid);

    detectExtrema(dog_pyramid, keypoints, contrast_threshold);
    assignOrientations(gauss_pyramid, keypoints);

    return keypoints;
}

/************************************************************************************************
 * Compute descriptors for keypoints.
 */
cv::Mat SIFTDetector::computeDescriptors(const cv::Mat &source_img, std::vector<cv::KeyPoint> &keypoints)
{
    cv::Mat descriptors;

    for (const auto &kp : keypoints)
    {
        cv::Mat descriptor = generateDescriptor(source_img, kp);
        descriptors.push_back(descriptor);
    }

    return descriptors;
}

void SIFTDetector::drawKeypoints(const cv::Mat& source_img, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& dest_img)
{
    cv::drawKeypoints(source_img, keypoints, dest_img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    for (const auto& kp : keypoints) {
        int radius = cvRound(kp.size * 5);
        cv::circle(dest_img, kp.pt, radius, cv::Scalar(255, 255, 255), 2);
    }
}