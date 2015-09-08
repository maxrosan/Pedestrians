
#include "functions.h"

std::vector<mpf_class> meanPositive;
std::vector<mpf_class> standPositive;
std::vector<mpf_class> meanNegative;
std::vector<mpf_class> standNegative;

int main(int argc, char** argv) {

    int y, x, a, j;
    unsigned int numPos, numNeg;
    cv::Mat viewer = cv::Mat(PERSON_WINDOW_H, PERSON_WINDOW_W, CV_8U,
            cv::Scalar(0));
    
    cv::Mat viewerNegative = cv::Mat(PERSON_WINDOW_H, PERSON_WINDOW_W, CV_8U,
            cv::Scalar(0));    
    
    float theta;

    loadValuesBayes(meanPositive, standPositive, meanNegative, standNegative, numPos, numNeg);
    
    j = 0;
    
    for (y = 0; y < (viewer.size().height / SIZE_WINDOW_HOG); y++) {
        for (x = 0; x < (viewer.size().width / SIZE_WINDOW_HOG); x++) {
            for (a = 0; a < SIZE_HOG; a++) {
                
                if (meanPositive[j] > 0.3) {

                    theta = (a * 180.) / SIZE_HOG;

                    cv::line(viewer,
                            cv::Point(x * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * cos(theta * M_PI / 180.),
                            y * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * sin(theta * M_PI / 180.)),
                            cv::Point(x * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * cos((theta + 180) * M_PI / 180.),
                            y * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * sin((theta + 180) * M_PI / 180.)),
                            cv::Scalar(round(255 * meanPositive[j].get_d())));
                }
                
                j++;
            }
        }
    }
    
    j = 0;
    
    for (y = 0; y < (viewer.size().height / SIZE_WINDOW_HOG); y++) {
        for (x = 0; x < (viewer.size().width / SIZE_WINDOW_HOG); x++) {
            for (a = 0; a < SIZE_HOG; a++) {
                
                if (meanNegative[j] > 0.3) {

                    theta = (a * 180.) / SIZE_HOG;

                    cv::line(viewerNegative,
                            cv::Point(x * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * cos(theta * M_PI / 180.),
                            y * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * sin(theta * M_PI / 180.)),
                            cv::Point(x * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * cos((theta + 180) * M_PI / 180.),
                            y * SIZE_WINDOW_HOG + SIZE_WINDOW_HOG * sin((theta + 180) * M_PI / 180.)),
                            cv::Scalar(round(255 * meanPositive[j].get_d())));
                }
                
                j++;
            }
        }
    }    
    
    int factor = 4;
    
    cv::resize(viewer, viewer, 
            cv::Size(viewer.size().width * factor, viewer.size().height * factor));
    cv::resize(viewerNegative, viewerNegative, 
            cv::Size(viewerNegative.size().width * factor, viewerNegative.size().height * factor));

    cv::imshow("result+", viewer);
    cv::imshow("result-", viewerNegative);

    while ((cv::waitKey(1) & 0xff) != 'q') {
    }


    return 0;
}

