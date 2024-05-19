#include "header/EdgeDetector.h"
#include "header/Filter.h"

int main(int argc, char** argv){
    EdgeDetector *edge_detector = new EdgeDetector();

    cv::Mat gray_img;
    cv::Mat filtered_img;
    cv::Mat dest_img;
    bool filtered = false;

    cv::Mat source_img = imread(argv[1], cv::IMREAD_COLOR);
    if (!source_img.data) {
        std::cout << "Image not found (wrong path) !";
        return 0;
    }

    /* ---------- Filter ---------- */
    if (argc == 4) {

    }
}
