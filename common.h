
#include <iostream>
#include <cstdlib>
#include <memory.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>
#include <vector>
#include <unistd.h>
#include <algorithm>
#include <pthread.h>
#include <dirent.h>
#include <fstream>
#include <queue>
#include <gmp.h>
#include <gmpxx.h>
#include <list>
#include <sys/stat.h>
#include <sys/types.h>

#define SIZE_HOG 8
#define SIZE_WINDOW_HOG 8
#define PERSON_WINDOW_W 64
#define PERSON_WINDOW_H 128
#define DEBUG 0
#define DEBUG_THREAD 0
#define N_THREADS 8
#define FACTOR 200
#define FACTOR_VERIFY "1.e30"
#define MAX_SIZE 341.0
#define THRESH_TRAIN 130.f
#define THRESH_CLASS 90.f
//#define THRESH 80.f
