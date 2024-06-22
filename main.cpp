#include "EdgeDetector.h"
#include "Filterer.h"
#include "CornerDetector.h"

int main(int argc, char **argv)
{
    EdgeDetector *edge_detector = new EdgeDetector();
    Filter *filter = new Filter();
    CornerDetector *corner_detector = new CornerDetector();

    cv::Mat gray_img;
    cv::Mat filtered_img;
    cv::Mat dest_img;
    bool filtered = false;

    cv::Mat source_img = imread(argv[2], cv::IMREAD_COLOR);
    if (!source_img.data)
    {
        std::cout << "\n Image not found (wrong path) !";
        std::cout << "\n Path: " << argv[2];
        return 0;
    }

    bool success = 0;

    if (argc == 5)
    {
        if (str_compare(argv[1], "-avg"))
        {
            success = filter->avg_filter(source_img, dest_img, char_2_int(argv, 4));
        }
        else if (str_compare(argv[1], "-median"))
        {
            success = filter->median_filter(source_img, dest_img, char_2_int(argv, 4));
        }
        else if (str_compare(argv[1], "-gau"))
        {
            success = filter->gaussian_filter(source_img, dest_img, char_2_int(argv, 4));
        }
        else
        {
            std::cout << "\n Undefined operation!";
        }
    }
    else if (argc == 4)
    {
        if (str_compare(argv[1], "-sobel"))
        {
            if (source_img.channels() == 3)
            {
                cv::Mat gray_img;
                filter->rgb2gray_filter(source_img, gray_img);
                filter->bilateral_filter(gray_img, filtered_img, 5, 10);
            }
            else
            {
                filter->bilateral_filter(source_img, filtered_img, 5, 10);
            }
            success = edge_detector->detect_by_sobel(filtered_img, dest_img);
        }
        else if (str_compare(argv[1], "-laplace"))
        {
            if (source_img.channels() == 3)
            {
                cv::Mat gray_img;
                filter->rgb2gray_filter(source_img, gray_img);
                filter->gaussian_filter(gray_img, filtered_img, 3);
            }
            else
            {
                filter->gaussian_filter(source_img, filtered_img, 3);
            }
            success = edge_detector->detect_by_laplace(filtered_img, dest_img);
        }
        else if (str_compare(argv[1], "-rgb2gray"))
        {
            success = filter->rgb2gray_filter(source_img, dest_img);
        }
        else
            std::cout << "\n Undefined operation!";
    }
    else if (argc == 6)
    {
        if (str_compare(argv[1], "-canny"))
        {
            // if (source_img.channels() == 3)
            // {
            //     cv::Mat gray_img;
            //     filter->rgb2gray_filter(source_img, gray_img);
            //     filter->gaussian_filter(gray_img, filtered_img, 5);
            // }
            // else
            // {
            //     filter->gaussian_filter(source_img, filtered_img, 5);
            // }
            success = edge_detector->detect_by_canny(source_img, dest_img, char_2_int(argv, 4), char_2_int(argv, 5));
        }
        else if (str_compare(argv[1], "-bil"))
        {
            success = filter->bilateral_filter(source_img, dest_img, char_2_int(argv, 4), char_2_int(argv, 5));
        }
    }
    else if (argc == 7)
    {
        if (str_compare(argv[1], "-harris"))
        {
            success = corner_detector->detect_corner_Harris(source_img, dest_img, char_2_int(argv, 4), char_2_int(argv, 5), char_2_double(argv, 6));
        }
    }
    else
    {
        std::cout << "\n Wrong input!";
        return 0;
    }

    if (success)
    {
        imshow("Source image", source_img);
        imshow("Processed image", dest_img);
        imwrite(argv[3], dest_img);
    }
    else
        std::cout << "\n Something went wrong!";

    cv::waitKey(0);
    return 0;
}
