#include "functions.h"

void pedestriansDetectionTrainBayes(std::vector<cv::Mat> &histograms,
        mpf_class* sum, mpf_class* sqSum, unsigned int &number) {

    int x, y, a, b, angle, i;
    int wHogPerson = (PERSON_WINDOW_W) / SIZE_WINDOW_HOG, hHogPerson = (PERSON_WINDOW_H) / SIZE_WINDOW_HOG;
    mpf_class value, totalSum;
    std::vector<float> descriptor(1024);

    std::cout << "training " << histograms[0].size() << std::endl;

    for (y = 0; y < histograms[0].size().height; y += hHogPerson) {
        for (x = 0; x < histograms[0].size().width; x += wHogPerson) {

            if ((y + hHogPerson) > histograms[0].size().height || (x + wHogPerson) > histograms[0].size().width) {
                continue;
            }

            descriptor.clear();

            for (b = 0; b < hHogPerson; b++) {
                for (a = 0; a < wHogPerson; a++) {
                    for (angle = 0; angle < SIZE_HOG; angle++) {
                        descriptor.push_back(histograms[angle].at<float>(y + b, x + a));
                    }
                }
            }

            totalSum = 0.;

            for (i = 0; i < descriptor.size(); i++) {
                value = mpf_class(descriptor[i]);
                sum[i] = sum[i] + value;
                sqSum[i] = sqSum[i] + (value * value);

                totalSum += value;
            }

            if (totalSum > 0.) {
                number++;
            }

        }
    }

}

void pedestriansDetectionBayes(std::vector<cv::Mat> &histograms, std::vector<cv::Point>& pedestrians,
        std::vector<mpf_class> &meanPositive, std::vector<mpf_class> &standPositive,
        std::vector<mpf_class> &meanNegative, std::vector<mpf_class> &standNegative,
        std::vector<cv::Point> &real, unsigned int &numPositive, unsigned int &numNegative) {

    int x, y, a, b, angle, i;
    int wHogPerson = (PERSON_WINDOW_W) / SIZE_WINDOW_HOG, hHogPerson = (PERSON_WINDOW_H) / SIZE_WINDOW_HOG;
    //mpf_class maxValue = 0., div;
    //std::vector<std::pair<mpf_class, cv::Point> > points;

    #pragma omp parallel for private (x, y, a, b, angle)
    for (i = 0; i < pedestrians.size(); i++) {

        x = pedestrians[i].x / SIZE_WINDOW_HOG;
        y = pedestrians[i].y / SIZE_WINDOW_HOG;

        //std::cout << histograms[0].size() << std::endl;

        std::vector<mpf_class> descriptor;
        mpf_class resultPositive, resultNegative, maxValue, div;

        for (b = 0; b < hHogPerson; b++) {
            for (a = 0; a < wHogPerson; a++) {
                for (angle = 0; angle < SIZE_HOG; angle++) {
                    descriptor.push_back(mpf_class(histograms[angle].at<float>(y + b, x + a)));
                }
            }
        }

        calculatePropability(meanPositive,
                standPositive,
                descriptor,
                resultPositive);

        resultPositive *= ((double) numPositive) / ((double) numPositive + numNegative);

        calculatePropability(meanNegative,
                standNegative,
                descriptor,
                resultNegative);

        resultNegative *= ((double) numNegative) / ((double) numPositive + numNegative);

#if DEBUG
        std::cout << cv::Point(x * SIZE_WINDOW_HOG, y * SIZE_WINDOW_HOG) << " ";
        std::cout << resultPositive << " " << resultNegative << std::endl;
#endif
        //std::cout << resultPositive / resultNegative << std::endl;

        div = (resultPositive / resultNegative);

        if (div >= mpf_class(FACTOR_VERIFY)) {
            //std::cout << (resultPositive / resultNegative) << std::endl;
            //points.push_back(
            //    std::make_pair(div, cv::Point(x * SIZE_WINDOW_HOG, y * SIZE_WINDOW_HOG)));
            #pragma omp critical(real)
            {
                real.push_back(cv::Point(x * SIZE_WINDOW_HOG, y * SIZE_WINDOW_HOG));
            }

            if (div > maxValue) {
                maxValue = div;
            }

#if DEBUG            
            std::cout << "OK" << std::endl;
#endif
        }
#if DEBUG            
        else {
            std::cout << "FAIL " << div << std::endl;
        }
#endif        
    }

#if 0
    for (i = 0; i < points.size(); i++) {
        div = points[i].first / maxValue;

        std::cout << div << std::endl;

        if (div > mpf_class(std::string("1e-7"))) {
            real.push_back(points[i].second);
        }

    }
#endif
}

void pedestriansInitialDetection(std::vector<cv::Mat> &histograms, std::vector<cv::Point>& pedestrians, std::vector<mpf_class**> &histogramsIntegral) {
    int a, x, y;
    int wSym = (PERSON_WINDOW_W) / SIZE_WINDOW_HOG, hSym = (PERSON_WINDOW_H) / SIZE_WINDOW_HOG;
    int xMin = -1, yMin = -1;

    histogramsIntegral.resize(SIZE_HOG, NULL);

    //#pragma omp parallel for
    for (a = 0; a < SIZE_HOG; a++) {
        calcIntegral<float>(histograms[a], histogramsIntegral[a]);
    }

    //#pragma omp parallel for private (x, a)
    for (y = 0; y < histograms[0].size().height; y++) {
        for (x = 0; x < histograms[0].size().width; x++) {

            mpf_class resultSym = 0., area = 0.;

            if ((x + wSym) >= histograms[0].size().width ||
                    (y + hSym) >= histograms[0].size().height) {
                continue;
            }

            for (a = 0; a < histogramsIntegral.size(); a++) {
                resultSym = resultSym + calcSym(histogramsIntegral[a], x, y, wSym, hSym);
                area = area + calcArea(histogramsIntegral[a], x, y, wSym, hSym);
            }

            resultSym = abs(resultSym);

            if (cmp(resultSym, 0.08) < 0 && area > 100.) {

                //std::cout << area << std::endl;

                //#pragma omp criticial
                {

                    pedestrians.push_back(cv::Point(x * SIZE_WINDOW_HOG, y * SIZE_WINDOW_HOG));

                }

            }

        }
    }
}

void pedestrianPreProcess(cv::Mat &inputImage, cv::Mat &angle, cv::Mat &scalar,
        std::vector<cv::Mat> &histograms, float thresh) {

    int a, b, x, y;
    float maxValue, val;
    double theta;

    histograms.resize(SIZE_HOG);

    assert(angle.empty() == false);
    assert(scalar.empty() == false);

    for (b = 0; b < SIZE_HOG; b++) {
        histograms[b] = cv::Mat(inputImage.size().height / SIZE_WINDOW_HOG, inputImage.size().width / SIZE_WINDOW_HOG, CV_32F, cv::Scalar(0));
    }

    #pragma omp parallel for private(a, x, y)
    for (b = 0; b < (inputImage.size().height - SIZE_WINDOW_HOG); b += (SIZE_WINDOW_HOG >> 1)) {
        for (a = 0; a < (inputImage.size().width - SIZE_WINDOW_HOG); a += (SIZE_WINDOW_HOG >> 1)) {

            for (y = 0; y < SIZE_WINDOW_HOG; y++) {
                for (x = 0; x < SIZE_WINDOW_HOG; x++) {
                    if (scalar.at<float>(y + b, x + a) > thresh) {

                        #pragma omp critical(histogram)
                        {
                            histograms[(angle.at<int>(y + b, x + a) * SIZE_HOG) / 180].at<float>(b / SIZE_WINDOW_HOG, a / SIZE_WINDOW_HOG) +=
                                gaussianWeight(x, y, PERSON_WINDOW_W, PERSON_WINDOW_H);
                        }

                    }
                }
            }

        }
    }

    //#pragma omp parallel for private(x, a, b, maxValue, theta)
    for (y = 0; y < histograms[0].size().height; y++) {
        for (x = 0; x < histograms[0].size().width; x++) {

            maxValue = 1e-5;

            for (a = 0; a < SIZE_HOG; a++) {
                val = fabs(histograms[a].at<float>(y, x));
                maxValue = std::max(val, maxValue);
            }

            if (maxValue > 1e-5) {

                for (a = 0; a < SIZE_HOG; a++) {
                    histograms[a].at<float>(y, x) /= maxValue;
                }

            } else {

                for (a = 0; a < SIZE_HOG; a++) {
                    histograms[a].at<float>(y, x) = 0.;
                }

            }

        }
    }

}

template <typename T>
void calcIntegral(cv::Mat &im, mpf_class** &integral) {

    integral = new mpf_class*[im.size().height + 1];

    for (int y = 0; y < im.size().height; y++) {

        integral[y] = new mpf_class[im.size().width + 1];

        for (int x = 0; x < im.size().width; x++) {

            integral[y][x] = mpf_class(0., PRECISION);

            if (y > 0 && x > 0) {

                /*std::cout << std::endl;
                std::cout << im.size() << " " << y << " " << x << std::endl;
                std::cout << im.at<T>(y, x) << std::endl;
                std::cout << integral[y][x - 1] << std::endl;
                std::cout << integral[y - 1][x] << std::endl;
                std::cout << integral[y - 1][x - 1] << std::endl;*/

                integral[y][x] = mpf_class(im.at<T>(y, x)) +
                        integral[y][x - 1] +
                        integral[y - 1][x] -
                        integral[y - 1][x - 1];
            }

        }
    }

}

mpf_class calcArea(mpf_class **integral, int a, int b, int w, int h) {

    mpf_class white(0.);

    white = integral[b + h][a + w] + integral[b][a]
            - integral[b][a + w] - integral[b + h][a];

    return (white);
}

mpf_class calcSym(mpf_class **integral, int a, int b, int w, int h) {

    mpf_class white(0.);
    mpf_class black(0.);

    white = integral[b + h][a + w / 2] + integral[b][a]
            - integral[b][a + w / 2] - integral[b + h][a];

    black = integral[b + h][a + w] + integral[b][a + w / 2]
            - integral[b][a + w] - integral[b + h][a + w / 2];


    return (white - black) / ((double) w * h);
}

void calculateGradient(cv::Mat &inputImage, cv::Mat &angle, cv::Mat &scalar) {

    std::vector<cv::Mat> channels;
    int a, b, y, x;
    cv::Mat angle_B, angle_G, angle_R,
            scalar_B, scalar_G, scalar_R;
    std::vector<int> descriptor;

    assert(inputImage.empty() == false);

    if (inputImage.channels() >= 3) {

        cv::split(inputImage, channels);

        #pragma omp parallel for
        for (b = 0; b < 3; b++) {
            if (b == 0) calculateHoG(channels[0], angle_B, scalar_B);
            else if (b == 1) calculateHoG(channels[1], angle_G, scalar_G);
            else calculateHoG(channels[2], angle_R, scalar_R);
        }

        angle = cv::Mat(inputImage.size().height, inputImage.size().width, CV_32S,
                cv::Scalar(0));

        scalar = cv::Mat(inputImage.size().height, inputImage.size().width, CV_32F,
                cv::Scalar(0));

        #pragma omp parallel for private(a)
        for (b = 0; b < inputImage.size().height; b++) {
            for (a = 0; a < inputImage.size().width; a++) {
                if (2 * scalar_B.at<float>(b, a) > (scalar_G.at<float>(b, a) + scalar_R.at<float>(b, a))) {
                    angle.at<int>(b, a) = angle_B.at<int>(b, a);
                    scalar.at<float>(b, a) = scalar_B.at<float>(b, a);
                } else if (2 * scalar_G.at<float>(b, a) > (scalar_B.at<float>(b, a) + scalar_R.at<float>(b, a))) {
                    angle.at<int>(b, a) = angle_G.at<int>(b, a);
                    scalar.at<float>(b, a) = scalar_G.at<float>(b, a);
                } else {
                    angle.at<int>(b, a) = angle_R.at<int>(b, a);
                    scalar.at<float>(b, a) = scalar_R.at<float>(b, a);
                }
            }
        }

        angle_R.release();
        angle_G.release();
        angle_B.release();
        scalar_R.release();
        scalar_G.release();
        scalar_B.release();

    } else {
        calculateHoG(inputImage, angle, scalar);
    }

}

void calculateHoG(cv::Mat &input, cv::Mat &angle, cv::Mat &scalar) {

    int a, b, degree;
    float m, theta, mxx, myy;
    cv::Mat Dxx, Dyy;

    angle = cv::Mat(input.size().height, input.size().width, CV_32S,
            cv::Scalar(0));

    scalar = cv::Mat(input.size().height, input.size().width, CV_32F,
            cv::Scalar(0));

    Dxx = cv::Mat(input.size().height, input.size().width, CV_32F,
            cv::Scalar(0));

    Dyy = cv::Mat(input.size().height, input.size().width, CV_32F,
            cv::Scalar(0));

    cv::Sobel(input, Dxx, CV_32F, 2, 0);
    cv::Sobel(input, Dyy, CV_32F, 0, 2);

    for (b = 0; b < input.size().height; b++) {
        for (a = 0; a < input.size().width; a++) {
            mxx = Dxx.at<float>(b, a);
            myy = Dyy.at<float>(b, a);
            m = sqrt(mxx * mxx + myy * myy);

            if (std::abs(m) < 1e-6f) {
                continue;
            }

            if (std::abs(mxx) < 1e-6f)
                theta = (myy < 0.f ? (2. * M_PI - M_PI / 2.) : (M_PI / 2.));
            else
                theta = atan(myy / mxx);

            theta *= (360. / (2. * M_PI));

            degree = ((int) round(theta) + 360) % 180;

            angle.at<int>(b, a) = degree;
            scalar.at<float>(b, a) = m;
        }
    }

}

double gaussianWeight(int xIndex, int yIndex, int W, int H) {

    static const double standX = W * 8, standY = H * 30, mean = 0;
    //static const double standX = W * 4, standY = H * 15, mean = 0;
    static const double a = 1.;

    double x = (double) (xIndex - W / 2);
    double y = (double) (yIndex - H / 2);
    double func = a * (1. / exp(
            x * x / (2. * standX) + y * y / (2. * standY)));

    return func;

}

void readDirectory(std::string folder,
        void (*functionFilePositive)(std::string),
        void (*functionFileNegative)(std::string)) {

    DIR *directory;
    struct dirent *entry;

    directory = opendir(folder.c_str());
    assert(directory != NULL);

    entry = readdir(directory);

    if (entry == NULL) {
        std::cerr << "Failed to read folder" << std::endl;
    }

    while (entry != NULL) {

        std::cout << "read " << entry->d_name << std::endl;

        if (strncmp(entry->d_name, "ap", 2) == 0) {

            functionFilePositive(folder + "/" + std::string(entry->d_name));

        } else if (strncmp(entry->d_name, "an", 2) == 0) {

            functionFileNegative(folder + "/" + std::string(entry->d_name));

        }

        entry = readdir(directory);
    }

}

void readDirectoryInput(std::string folder,
        void (*functionFile)(std::string)) {

    DIR *directory;
    struct dirent *entry;

    directory = opendir(folder.c_str());
    assert(directory != NULL);

    entry = readdir(directory);

    if (entry == NULL) {
        std::cerr << "Failed to read folder" << std::endl;
    }

    while (entry != NULL) {

        if (strncmp(entry->d_name, "qx", 2) == 0) {

            functionFile(folder + "/" + std::string(entry->d_name));

        }

        entry = readdir(directory);
    }

}

void calculateValues(mpf_class* sumPositive, mpf_class* sqSumPositive,
        mpf_class* sumNegative, mpf_class* sqSumNegative,
        unsigned int numberOfPositives, unsigned int numberOfNegatives) {

    int i;
    mpf_class mean, stand;
    std::ofstream outputFile;

    outputFile.open("values.txt");

    outputFile << numberOfPositives;

    outputFile.precision(15);

    for (i = 0; i < 1024; i++) {
        mean = sumPositive[i] / numberOfPositives;
        //std::cout << sqSumPositive[i] << " " << sumPositive[i] << " " << sqSumPositive[i] - (sumPositive[i] * sumPositive[i]) << std::endl;
        //stand = sqrt((sqSumPositive[i] - (sumPositive[i] * sumPositive[i]) / mpf_class(numberOfPositives)) / mpf_class(numberOfPositives));
        stand = sqrt(sqSumPositive[i] / numberOfPositives - mean * mean);
        outputFile << mean << " " << stand << std::endl;
    }

    outputFile << numberOfNegatives;

    for (i = 0; i < 1024; i++) {
        mean = sumNegative[i] / numberOfNegatives;
        stand = sqrt(sqSumNegative[i] / numberOfNegatives - mean * mean);
        //stand = sqrt((sqSumNegative[i] - (sumNegative[i] * sumNegative[i]) / mpf_class(numberOfNegatives)) / mpf_class(numberOfNegatives));
        //outputFile << mean << " " << stand << std::endl;
        outputFile << mean << " " << stand << std::endl;
    }

    outputFile.close();
}

mpf_class expMP(mpf_class ex, int num, int precision) {

    int i;
    mpf_class result = mpf_class(1., precision);
    mpf_class x = ex;
    mpf_class factor = mpf_class(1., precision);

    assert(ex < mpf_class(1.));

    for (i = 1; i <= num; i++) {
        result += (x / factor);
        x *= x;
        factor *= mpf_class(i, precision);
    }

    return result;
}

void loadValuesBayes(std::vector<mpf_class> &meanPositive, std::vector<mpf_class> &standPositive,
        std::vector<mpf_class> &meanNegative, std::vector<mpf_class> &standNegative,
        unsigned int &numPositive, unsigned int &numNegative) {

    std::ifstream inputFile;
    mpf_class mean, stand;
    int i;

    inputFile.open("values.txt");

    inputFile >> numPositive;

    for (i = 0; i < 1024; i++) {
        inputFile >> mean >> stand;

        meanPositive.push_back(mean);
        standPositive.push_back(stand);
    }

    inputFile >> numNegative;

#if DEBUG
    std::cout << "+ " << meanPositive.size() << " " << standPositive.size() << std::endl;
#endif

    for (i = 0; i < 1024; i++) {
        inputFile >> mean >> stand;

        meanNegative.push_back(mean);
        standNegative.push_back(stand);
    }

#if DEBUG
    std::cout << "- " << meanNegative.size() << " " << standNegative.size() << std::endl;
#endif

    inputFile.close();

}

static mpf_class expT(mpf_class x, unsigned int n) {

    mpf_class result(1., 128), one(1.), nMP(n);
    mpf_class factor = (one + (x / nMP));
    int i = 0;

    while (n > 1) {

        i++;

        if ((n & 1) == 0) {
            factor *= factor;
            n = n >> 1;
        } else {
            result = factor * result;
            factor *= factor;
            n = (n - 1) >> 1;
        }
    }

    return result * factor;
}

mpf_class expMP(mpf_class x) {
    return expT(x, 1e7);
}

void calculatePropability(std::vector<mpf_class> &mean,
        std::vector<mpf_class> &stand,
        std::vector<mpf_class> &descriptor,
        mpf_class &result) {

    int i;

    
    DEC_FLOAT(constantNumber);
    FIX_PREC(result);

    result = 1.;
    constantNumber = 2. * M_PI;

    #pragma omp parallel for
    for (i = 0; i < descriptor.size(); i++) {
        
        DEC_FLOAT(doubleStand);
        DEC_FLOAT(doubleMean);
        DEC_FLOAT(power);
        DEC_FLOAT(div);
        DEC_FLOAT(prob);

        if (stand[i] > mpf_class(std::string("0.000000001"))) {

            doubleMean = (descriptor[i] - mean[i]);
            doubleStand = stand[i] * stand[i];

            power = (doubleMean * doubleMean) / (2. * doubleStand);
            div = sqrt(constantNumber * doubleStand);
            div *= expMP(power);
            prob = 1. / div;

            #pragma omp critical(result)
            {
                result = result * prob;
            }

        }

    }

}

void postProcess(std::vector<cv::Rect> regions, std::vector<cv::Rect> &result) {

    int i, j;
    float intersectArea;
    cv::Rect rect;
    std::vector<bool> validRects(regions.size(), true);

    using cv::operator&;

    for (i = 0; i < regions.size(); i++) {

        if (!validRects[i]) {
            continue;
        }

        for (j = 0; j < regions.size(); j++) {

            if (i == j || !validRects[j]) {
                continue;
            }

            rect = regions[i] & regions[j];
            intersectArea = ((float) rect.area());
            intersectArea /= ((float) std::max(regions[i].area(), regions[j].area()));

            if (intersectArea > 0.10 ||
                    regions[i].contains(cv::Point(regions[j].x, regions[j].y)) &&
                    regions[i].contains(cv::Point(regions[j].x + regions[j].width, regions[j].y + regions[j].height))) {
                if (regions[i].area() > regions[j].area()) {
                    validRects[j] = false;
                } else {
                    validRects[i] = false;
                    j = regions.size() + 1;
                }
            } else {
#if DEBUG
                std::cout << intersectArea << std::endl;
#endif
            }

        }
    }

    for (i = 0; i < regions.size(); i++) {
        if (validRects[i]) {
            rect = regions[i];
            //cv::rectangle(image, rect, cv::Scalar(0, 255, 0), 1);
            result.push_back(rect);
        }
    }

}