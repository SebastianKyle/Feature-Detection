#include "Filterer.h"

Filter::Filter()
{
}

Filter::~Filter()
{
}

int Filter::rgb2gray_filter(const cv::Mat& source_img, cv::Mat& dest_img) {
    if (!source_img.data)
        return 0;

    cv::Mat output = cv::Mat::zeros(source_img.size(), CV_8UC1);
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            cv::Vec3b pixel = source_img.at<cv::Vec3b>(y, x);
            int gray = 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];
            output.at<uchar>(y, x) = cv::saturate_cast<uchar>(gray);
        }
    }

    dest_img = output.clone();
    return 1;
}

int Filter::avg_filter(const cv::Mat &source_img, cv::Mat &dest_img, int k)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);

            if (source_img.channels() == 3)
            {
                cv::Vec3i convolve(0, 0, 0);

                // Border: pad with reflection
                for (int yi = y - quarter_side; yi <= y + quarter_side; yi++)
                {
                    for (int xi = x - quarter_side; xi <= x + quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(yi));
                        int x_coord = std::min(width - 1, abs(xi));
                        convolve += source_img.at<cv::Vec3b>(y_coord, x_coord);
                    }
                }

                convolve = convolve / (k * k);

                cv::Vec3b convolve_b;
                convolve_b[0] = cv::saturate_cast<uchar>(convolve[0]);
                convolve_b[1] = cv::saturate_cast<uchar>(convolve[1]);
                convolve_b[2] = cv::saturate_cast<uchar>(convolve[2]);

                output.at<cv::Vec3b>(y, x) = convolve_b;
            }
            else
            {
                int convolve = 0;

                // Border: pad with reflection
                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));
                        convolve += source_img.at<uchar>(y_coord, x_coord);
                    }
                }

                convolve = int(convolve / (k * k));

                output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve);
            }
        }
    }

    dest_img = output.clone();
    return 1;
}

int Filter::median_filter(const cv::Mat &source_img, cv::Mat &dest_img, int k)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);

            if (source_img.channels() == 3)
            {
                std::vector<int> medians[3];

                // Border: pad with reflection
                for (int yi = y - quarter_side; yi <= y + quarter_side; yi++)
                {
                    for (int xi = x - quarter_side; xi <= x + quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(yi));
                        int x_coord = std::min(width - 1, abs(xi));

                        cv::Vec3b pixel = source_img.at<cv::Vec3b>(y_coord, x_coord);
                        for (int c = 0; c < 3; c++)
                        {
                            medians[c].push_back(pixel[c]);
                        }
                    }
                }

                cv::Vec3b median;
                for (int c = 0; c < 3; c++)
                {
                    std::sort(medians[c].begin(), medians[c].end());
                    median[c] = medians[c][medians[c].size() / 2];
                }

                output.at<cv::Vec3b>(y, x) = median;
            }
            else
            {
                std::vector<int> medians;

                // Border: pad with reflection
                for (int yi = y - quarter_side; yi <= y + quarter_side; yi++)
                {
                    for (int xi = x - quarter_side; xi <= x + quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(yi));
                        int x_coord = std::min(width - 1, abs(xi));

                        medians.push_back(source_img.at<uchar>(y_coord, x_coord));
                    }
                }

                std::sort(medians.begin(), medians.end());
                uchar median = medians[medians.size() / 2];

                output.at<uchar>(y, x) = median;
            }
        }
    }

    dest_img = output.clone();
    return 1;
}

int Filter::gaussian_filter(const cv::Mat &source_img, cv::Mat &dest_img, int k)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);
            double w = 0;
            int sigma = int(k / 6);
            sigma = sigma == 0 ? 1 : sigma;

            if (source_img.channels() == 3)
            {
                cv::Vec3i convolve(0, 0, 0);

                // Border: pad with reflection
                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));

                        cv::Vec3b pixel = source_img.at<cv::Vec3b>(y_coord, x_coord);

                        double distance = yi * yi + xi * xi;
                        double weight = (1 / (2 * 3.14 * sigma * sigma)) * exp(-distance / (2 * sigma * sigma));

                        for (int c = 0; c < 3; c++)
                        {
                            convolve[c] += pixel[c] * weight;
                        }

                        w += weight;
                    }
                }

                cv::Vec3b convolve_b;
                convolve_b[0] = cv::saturate_cast<uchar>(convolve[0] / w);
                convolve_b[1] = cv::saturate_cast<uchar>(convolve[1] / w);
                convolve_b[2] = cv::saturate_cast<uchar>(convolve[2] / w);

                output.at<cv::Vec3b>(y, x) = convolve_b;
            }
            else
            {
                int convolve = 0;

                // Border: pad with reflection
                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));

                        uchar pixel = source_img.at<uchar>(y_coord, x_coord);

                        double distance = yi * yi + xi * xi;
                        double weight = (1 / (2 * 3.14 * sigma * sigma)) * exp(-distance / (2 * sigma * sigma));

                        convolve += pixel * weight;
                        w += weight;
                    }
                }

                output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve / w);
            }
        }
    }

    dest_img = output.clone();
    return 1;
}

int Filter::bilateral_filter(const cv::Mat &source_img, cv::Mat &dest_img, int k, int sigma_b)
{
    if (!source_img.data)
        return 0;

    cv::Mat output = source_img.clone();
    int width = source_img.cols, height = source_img.rows;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Expect k = 2t + 1
            int quarter_side = int((k - 1) / 2);
            int sigma = int(k / 6);
            sigma = sigma == 0 ? 1 : sigma;

            if (source_img.channels() == 3)
            {
                cv::Vec3i convolve(0, 0, 0);
                cv::Vec3d wsb(0, 0, 0);

                // Border: pad with reflection
                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));

                        cv::Vec3b pixel = source_img.at<cv::Vec3b>(y_coord, x_coord);
                        for (int c = 0; c < 3; c++)
                        {
                            // Spatial gaussian
                            double distance = yi * yi + xi * xi;
                            double n_sigma_s = exp(-distance / (2 * sigma * sigma));

                            // Brightness gaussian
                            double brightness_diff = (pixel[c] - source_img.at<cv::Vec3b>(y, x)[c]) * (pixel[c] - source_img.at<cv::Vec3b>(y, x)[c]);
                            double n_sigma_b = exp(-brightness_diff / (2 * sigma_b * sigma_b));

                            convolve[c] += pixel[c] * n_sigma_s * n_sigma_b;
                            wsb[c] += n_sigma_s * n_sigma_b;
                        }
                    }
                }

                cv::Vec3b convolve_b;
                convolve_b[0] = cv::saturate_cast<uchar>(convolve[0] / wsb[0]);
                convolve_b[1] = cv::saturate_cast<uchar>(convolve[1] / wsb[1]);
                convolve_b[2] = cv::saturate_cast<uchar>(convolve[2] / wsb[2]);

                output.at<cv::Vec3b>(y, x) = convolve_b;
            }
            else
            {
                int convolve = 0;
                double wsb = 0;

                // Border: pad with reflection
                for (int yi = -quarter_side; yi <= quarter_side; yi++)
                {
                    for (int xi = -quarter_side; xi <= quarter_side; xi++)
                    {
                        int y_coord = std::min(height - 1, abs(y + yi));
                        int x_coord = std::min(width - 1, abs(x + xi));

                        // Spatial gaussian
                        double distance_sq = yi * yi + xi * xi;
                        double n_sigma_s = exp(-distance_sq / (2 * 3.14 * sigma * sigma));

                        // Brightness gaussian
                        double brightness_diff_sq = (source_img.at<uchar>(y_coord, x_coord) - source_img.at<uchar>(y, x)) * (source_img.at<uchar>(y_coord, x_coord) - source_img.at<uchar>(y, x));
                        double n_sigma_b = exp(-brightness_diff_sq / (2 * sigma_b * sigma_b));

                        convolve += source_img.at<uchar>(y_coord, x_coord) * n_sigma_s * n_sigma_b;
                        wsb += n_sigma_s * n_sigma_b;
                    }
                }
                convolve = int(convolve / wsb);

                output.at<uchar>(y, x) = cv::saturate_cast<uchar>(convolve);
            }
        }
    }

    dest_img = output.clone();
    return 1;
}