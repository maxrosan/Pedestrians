
#include "common.h"
#include <omp.h>

//#define THRESH 150.f
#define THREAD_PRINT std::cout << "thread = " << omp_get_thread_num() << std::endl

#define PRECISION 512
#define DEC_FLOAT(NAME) mpf_class NAME (0., PRECISION)
#define FIX_PREC(NAME) NAME.set_prec(PRECISION)

mpf_class calcArea(mpf_class **integral, int a, int b, int w, int h);
mpf_class calcSym(mpf_class **integral, int a, int b, int w, int h);
template <typename T>
void calcIntegral(cv::Mat &im, mpf_class** &integral);
void calculateGradient(cv::Mat &inputImage, cv::Mat &angle, cv::Mat &scalar);
void pedestrianPreProcess(cv::Mat &inputImage, cv::Mat &angle, cv::Mat &scalar, std::vector<cv::Mat> &histograms,
        float tresh);
void pedestriansInitialDetection(std::vector<cv::Mat> &histograms, std::vector<cv::Point>& pedestrians, std::vector<mpf_class**> &histogramsIntegral);

void pedestriansDetectionBayes(std::vector<cv::Mat> &histograms, std::vector<cv::Point>& pedestrians,
        std::vector<mpf_class> &meanPositive, std::vector<mpf_class> &standPositive,
        std::vector<mpf_class> &meanNegative, std::vector<mpf_class> &standNegative,
        std::vector<cv::Point> &real, unsigned int &numPositive, unsigned int &numNegative);

void calculateHoG(cv::Mat &input, cv::Mat &angle, cv::Mat &scalar);
double gaussianWeight(int xIndex, int yIndex, int W, int H);
void readDirectory(std::string folder, void (*functionFilePositive)(std::string), void (*functionFileNegative)(std::string));
void pedestriansDetectionTrainBayes(std::vector<cv::Mat> &histograms, 
        mpf_class* sum, mpf_class* sumSq, unsigned int &number);
void calculateValues(mpf_class* sumPositive, mpf_class* sqSumPositive,
        mpf_class* sumNegative, mpf_class* sqSumNegative,
        unsigned int numberOfPositives, unsigned int numberOfNegatives);
void loadValuesBayes(std::vector<mpf_class> &meanPositive, std::vector<mpf_class> &standPositive,
        std::vector<mpf_class> &meanNegative, std::vector<mpf_class> &standNegative,
        unsigned int &numPositive, unsigned int &numNegative);
mpf_class expMP(mpf_class x);
void calculatePropability(std::vector<mpf_class> &mean,
        std::vector<mpf_class> &stand, 
        std::vector<mpf_class> &descriptor, 
        mpf_class &result);
void postProcess(std::vector<cv::Rect> regions, std::vector<cv::Rect> &result);
void readDirectoryInput(std::string folder,
        void (*functionFile)(std::string));