
#include "functions.h"

mpf_class *sumPositive, *sumSqPositive, *sumNegative, *sumSqNegative;
unsigned int numberOfPositive = 0, numberOfNegatives = 0;

void functionFilePositive(std::string fileName) {    
    cv::Mat image, angle, scalar;
    std::vector<cv::Mat> histograms;
    
    image = cv::imread(fileName);
    assert(image.empty() == false);
    
    calculateGradient(image, angle, scalar);
    pedestrianPreProcess(image, angle, scalar, histograms, THRESH_TRAIN);
    pedestriansDetectionTrainBayes(histograms, sumPositive, sumSqPositive, numberOfPositive);

}

void functionFileNegative(std::string fileName) {
    cv::Mat image, angle, scalar;
    std::vector<cv::Mat> histograms;
    
    image = cv::imread(fileName);
    assert(image.empty() == false);
    
    calculateGradient(image, angle, scalar);
    pedestrianPreProcess(image, angle, scalar, histograms, THRESH_TRAIN);
    pedestriansDetectionTrainBayes(histograms, sumNegative, sumSqNegative, numberOfNegatives);
}

int main(int argc, char **argv) {

  std::string folder;
  int i;

  omp_set_num_threads(8);

  //assert(argc == 2);

  //folder = std::string(argv[1]);
  
  folder = std::string("images");
  
  sumPositive = new mpf_class[1024];
  sumSqPositive = new mpf_class[1024];
  sumNegative = new mpf_class[1024];
  sumSqNegative = new mpf_class[1024];

  for (i = 0; i < 1024; i++) {
    sumPositive[i] = mpf_class(0.);
    sumSqPositive[i] = mpf_class(0.);
    sumNegative[i] = mpf_class(0.);
    sumSqNegative[i] = mpf_class(0.);
  }
  
  readDirectory(folder, functionFilePositive, functionFileNegative);
  calculateValues(sumPositive, sumSqPositive, sumNegative, sumSqNegative,
          numberOfPositive, numberOfNegatives);
  
  std::cout << numberOfPositive << " " << numberOfNegatives << std::endl;

  return EXIT_SUCCESS;
  
}