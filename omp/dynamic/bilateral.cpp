#include<opencv4/opencv2/opencv.hpp>
#include<omp.h>
#include<iostream>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace cv;

float distance(int x, int y, int i, int j) {
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) {
    return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));

}

void applyBilateralFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
    double iFiltered = 0;
    double wP = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;
    int i = 0, j = 0;

    for (i = 0; i < diameter; i++) {
        for (j = 0; j < diameter; j++) {
            neighbor_x = x - (half - i);
            neighbor_y = y - (half - j);
            double gi = gaussian(source.at<uchar>(neighbor_x, neighbor_y) - source.at<uchar>(x, y), sigmaI);
            double gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
            double w = gi * gs;
            iFiltered = iFiltered + source.at<uchar>(neighbor_x, neighbor_y) * w;
            wP = wP + w;
        }
    }
    iFiltered = iFiltered / wP;
    filteredImage.at<double>(x, y) = iFiltered;


}

Mat myBilateralFilter(Mat source, int diameter, double sigmaI, double sigmaS) {
    Mat filteredImage = Mat::zeros(source.rows, source.cols, CV_64F);
    int width = source.cols;
    int height = source.rows;

    int i = 1, j = 1;

#pragma omp parallel for collapse(2) schedule(runtime)
    for (i = 1; i < height - 1; i++) {
        for (j = 1; j < width - 1; j++) {
            applyBilateralFilter(source, filteredImage, i, j, diameter, sigmaI, sigmaS);
        }
    }

    return filteredImage;
}


int main(int argc, char **argv) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    Mat originalImage;
    originalImage = imread(argv[1], IMREAD_GRAYSCALE);
    int num_iterations = atoi(argv[2]);
    omp_set_num_threads(4);
    omp_set_schedule(omp_sched_dynamic, 4);
    if (!originalImage.data) {
        std::cout << "Image not found or unable to open" << std::endl;
        return -1;
    }

    int diameter = 3;
    double sigmaI = 12.0;
    double sigmaS = 16.0;
    Mat myFilteredImage;

    auto t3 = high_resolution_clock::now();

#pragma omp parallel for schedule(runtime)
    for (int i = 0; i < num_iterations; i++) {
        std::cout << "#";
        std::cout.flush();
        myFilteredImage = myBilateralFilter(originalImage, diameter, sigmaI, sigmaS);
    }
    printf("\n");
    auto t4 = high_resolution_clock::now();
    auto ms_int2 = duration_cast<milliseconds>(t4 - t3);
    std::cout << ms_int2.count() << "ms\n";

    imwrite("omp_bilateral_image.bmp", myFilteredImage);

    return 0;
}