#include<opencv4/opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;

float distance(int x, int y, int i, int j) {
    return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

double gaussian(float x, double sigma) {
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));

}

void applyBilateralFilter(Mat source, Mat filteredImage, int x, int y, int diameter, double sigmaI, double sigmaS) {
    double iFiltered = 0;
    double wP = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;

    for(int i = 0; i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
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
    Mat filteredImage = Mat::zeros(source.rows,source.cols,CV_64F);
    int width = source.cols;
    int height = source.rows;

    for(int i = 1; i < height - 1; i++) {
        for(int j = 1; j < width - 1; j++) {
            applyBilateralFilter(source, filteredImage, i, j, diameter, sigmaI, sigmaS);
        }
    }
    return filteredImage;
}


int main(int argc, char** argv) {
  
    Mat originalImage;
    originalImage = imread(argv[1], IMREAD_GRAYSCALE);
    int num_iterations = atoi(argv[2]);
    
    if(! originalImage.data ) {
      std::cout <<  "Image not found or unable to open" << std::endl ;
      return -1;
    }

    int diameter = 3;
    double sigmaI = 12.0;
    double sigmaS = 16.0;

    Mat filteredImageOpenCV, myFilteredImage;
    bilateralFilter(originalImage, filteredImageOpenCV, diameter, sigmaI, sigmaS);
    imwrite("./opencv_output_images/opencv_bilateral_Elaine.bmp", filteredImageOpenCV);

    printf("Average elapsed time for serial version of Bilateral Filter= %d secs\n", num_iterations);
    const clock_t begin_time = clock();
//    for (int i = 0; i < num_iterations; i++)
//    {
//        printf("Process= %d/100\n", i);
        myFilteredImage = myBilateralFilter(originalImage, diameter, sigmaI, sigmaS);
//    }
    std::cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC;
    printf("\nAverage elapsed time for serial version of Bilateral Filter= %d secs\n", num_iterations);

    imwrite("./amir_output_images/my_bilateral_Elaine.bmp", myFilteredImage);

    return 0;
}