
#include "functions.h"

char buff[512];

static inline void detectPedestrian(std::string imageFileName) {
    cv::Mat originalImage, finalImage;
    std::vector<mpf_class> meanPositive, standPositive, meanNegative, standNegative;
    std::vector<cv::Rect> rects, validRects;
    int i, scale, a, b;
    unsigned int numberOfPositives, numberOfNegatives;

    originalImage = cv::imread(imageFileName);

    assert(originalImage.empty() == false);

    loadValuesBayes(meanPositive, standPositive, meanNegative, standNegative,
            numberOfPositives, numberOfNegatives);

    //image = originalImage.clone();

    for (scale = 1; scale <= 2; scale *= 2) {

        std::vector<cv::Point> realPedestrians;
        cv::Mat image, angle, scalar;
        std::vector<cv::Point> pedestrians;
        std::vector<cv::Mat> histograms;
        std::vector<mpf_class**> histogramsIntegral;

#if DEBUG
        std::cout << "scale = " << scale << std::endl;
#endif

        cv::resize(originalImage,
                image,
                cv::Size(
                originalImage.size().width * scale,
                originalImage.size().height * scale));

        calculateGradient(image, angle, scalar);
        pedestrianPreProcess(image, angle, scalar, histograms, THRESH_CLASS);

        pedestriansInitialDetection(histograms, pedestrians, histogramsIntegral);
        pedestriansDetectionBayes(histograms, pedestrians,
                meanPositive, standPositive, meanNegative, standNegative,
                realPedestrians, numberOfPositives, numberOfNegatives);


        for (i = 0; i < realPedestrians.size(); i++) {

            //std::cout << realPedestrians[i] << std::endl;

            cv::Rect rect;

            rect.x = round(realPedestrians[i].x / ((double) scale));
            rect.y = round(realPedestrians[i].y / ((double) scale));
            rect.width = round(PERSON_WINDOW_W / ((double) scale));
            rect.height = round(PERSON_WINDOW_H / ((double) scale));

            /*cv::rectangle(initialImage,
                    cv::Rect(realPedestrians[i].x,
                    realPedestrians[i].y,
                    PERSON_WINDOW_W,
                    PERSON_WINDOW_H),
                    cv::Scalar(0, 0, 255));*/

            rects.push_back(rect);

#if DEBUG
            std::cout << rect << std::endl;
#endif

        }

    }

    postProcess(rects, validRects);
    rects = std::vector<cv::Rect>(validRects);
    validRects.clear();

    finalImage = originalImage.clone();

    for (i = 0; i < rects.size(); i++) {
        cv::rectangle(finalImage,
                rects[i],
                cv::Scalar(0, 0, 255));
    }
    
    imageFileName = std::string("saida/") +
            std::string(
                basename(imageFileName.replace(imageFileName.find("qx"), 2, "qp").c_str()));
    
    cv::imwrite(imageFileName, finalImage);
    
    originalImage.release();
    finalImage.release();    
    
}

int main(int argc, char **argv) {

    omp_set_num_threads(8);

    readDirectoryInput("images", detectPedestrian);

    //imageFileName = std::string(argv[1]);

    /*cv::imshow("initial-image", finalImage);

    while ((cv::waitKey(1) & 0xff) != 'q') {
    }*/

    return EXIT_SUCCESS;
}
