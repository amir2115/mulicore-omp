#include<opencv4/opencv2/opencv.hpp>
#include<omp.h>
#include<iostream>
#include <iomanip>
#include <chrono>
//#include"Timer.h"

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

//Mat myBilateralFilter(Mat source, int diameter, double sigmaI, double sigmaS) {
//    Mat filteredImage = Mat::zeros(source.rows, source.cols, CV_64F);
//    int width = source.cols;
//    int height = source.rows;
//
//    int i = 1, j = 1;
//    omp_set_num_threads(4);
//
//#pragma omp for private(i, j)
//    for (i = 1; i < height - 1; i++) {
//        // #pragma omp for
//        for (j = 1; j < width - 1; j++) {
//            applyBilateralFilter(source, filteredImage, i, j, diameter, sigmaI, sigmaS);
//        }
//    }
//
//    return filteredImage;
//}

Mat myBilateralFilter(Mat source, int diameter, double sigmaI, double sigmaS) {
    Mat filteredImage = Mat::zeros(source.rows, source.cols, CV_64F);
    int width = source.cols;
    int height = source.rows;

    int i = 1, j = 1;


#pragma omp parallel for collapse(2)
    for (i = 1; i < height - 1; i++) {
        // #pragma omp for
        for (j = 1; j < width - 1; j++) {
            applyBilateralFilter(source, filteredImage, i, j, diameter, sigmaI, sigmaS);
        }
    }

    return filteredImage;
}


int main(int argc, char **argv) {

    Mat originalImage;
    originalImage = imread(argv[1], IMREAD_GRAYSCALE);
    int num_iterations = atoi(argv[2]);
    omp_set_num_threads(4);
    if (!originalImage.data) {
        std::cout << "Image not found or unable to open" << std::endl;
        return -1;
    }

    int diameter = 3;
    double sigmaI = 12.0;
    double sigmaS = 16.0;

    Mat filteredImageOpenCV, myFilteredImage;
    bilateralFilter(originalImage, filteredImageOpenCV, diameter, sigmaI, sigmaS);
    imwrite("opencv_bilateral_image.bmp", filteredImageOpenCV);
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
//    Timer timer;
    float total_time;
//    const clock_t begin_time = clock();
    clock_t start, end;

    /* Recording the starting clock tick.*/
    start = clock();
#pragma omp parallel for
    for (int i = 0; i < num_iterations; i++) {
//        timer.start();
//        clock_t begin_time = clock();
        printf("Process= %d/100\n", i);
        myFilteredImage = myBilateralFilter(originalImage, diameter, sigmaI, sigmaS);
//        total_time += timer.elapsedTime();
//        total_time += float( clock () - begin_time );
    }
    end = clock();
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
//    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
//    double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
//    cout << "Time taken by program is : " << fixed
//         << time_taken << setprecision(10);
//    cout << " sec " << endl;

//    printf("\nAverage elapsed time for parallel version of Bilateral Filter= %f secs\n", total_time);
//    printf("\nAverage elapsed time for parallel version of Bilateral Filter= %df secs\n", num_iterations);

    imwrite("my_bilateral_image.bmp", myFilteredImage);

    return 0;
}