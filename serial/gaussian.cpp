#include<opencv4/opencv2/opencv.hpp>
#include<iostream>
#include<math.h>

using namespace std;
using namespace cv;

double gaussian(float x, double sigma) {
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * 3.14 * pow(sigma, 2));

}

void applyGaussianFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigma) {
    double iFiltered = 0;
    double wP = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;

    for(int i = 0; i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            neighbor_x = x - (half - i);
            neighbor_y = y - (half - j);
            double w = gaussian(source.at<uchar>(neighbor_x, neighbor_y) - source.at<uchar>(x, y), sigma);
            iFiltered = iFiltered + source.at<uchar>(neighbor_x, neighbor_y) * w;
            wP = wP + w;
        }
    }
    iFiltered = iFiltered / wP;
    filteredImage.at<double>(x, y) = iFiltered;

}

Mat myGaussianFilter(Mat source, int diameter, double sigma) {
    Mat filteredImage = Mat::zeros(source.rows,source.cols,CV_64F);
    int width = source.cols;
    int height = source.rows;

    for(int i = 2; i < height - 2; i++) {
        for(int j = 2; j < width - 2; j++) {
            applyGaussianFilter(source, filteredImage, i, j, diameter, sigma);
        }
    }
    return filteredImage;
}

int main(int argc, char** argv) {
  
    Mat originalImage;
    originalImage = imread(argv[1], IMREAD_GRAYSCALE);
    
    if(! originalImage.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
    }

    int diameter = 5;
    double sigma = 12.0;

    Mat filteredImageOpenCV;
    GaussianBlur(originalImage, filteredImageOpenCV, Size(diameter, diameter), sigma, sigma);
    
    cv::imwrite("opencv_gaussian_image.bmp", filteredImageOpenCV);

    Mat myFilteredImage = myGaussianFilter(originalImage, diameter, sigma);
    cv::imwrite("my_gaussian_image.bmp", myFilteredImage);

    return 0;
}