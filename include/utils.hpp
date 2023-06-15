#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const Scalar COLOR_BLUE(255, 0, 0);
const Scalar COLOR_GREEN(0, 255, 0);
const Scalar COLOR_RED(0, 0, 255);
const Scalar COLOR_WHITE(255, 255, 255);
const Scalar COLOR_YELLOW(0, 255, 255);
const Scalar COLOR_CYAN(255, 255, 0);
const Scalar COLOR_MAGENTA(255, 0, 255);

#define CONFIG_PATH (SOURCE_DIR "config.yaml")
#define LOAD_CONFIG(var) \
  do {                   \
    FileStorage config(CONFIG_PATH, FileStorage::READ);\
    auto prev = var;     \
    config[#var] >> var; \
    if (var == prev) throw std::runtime_error("Invalid config for variable \"" #var "\"");\
  } while(0);

#define RUNTIME(name, code)                                                                                                                \
  do {                                                                                                                                     \
    const int count = 1;                                                                                                                   \
    long sum = 0;                                                                                                                          \
    for (size_t _i = 0; _i < count; _i++) {                                                                                                \
      auto start = chrono::high_resolution_clock::now();                                                                                   \
      code;                                                                                                                                \
      auto stop = chrono::high_resolution_clock::now();                                                                                    \
      auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);                                                            \
      sum += duration.count();                                                                                                             \
    }                                                                                                                                      \
    float average = sum / count;                                                                                                           \
    if (average > 5 * 1000)                                                                                                                \
      std::printf(name " runtime: %ld us\n", (long)(average / 1000));                                                                      \
    else                                                                                                                                   \
      std::printf(name " runtime: %ld ns\n", (long)(average));                                                                             \
  } while (0)

template<typename F>
static inline int runtime_us(F && code) {
      auto start = chrono::high_resolution_clock::now();
      code();
      auto stop = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
      return duration.count();
}

const int circle_size = 2;
const int line_thickness = 1;

void draw_square(Mat &img, Point2f center, int half_width, Scalar color);
void draw_quad(Mat &img, std::vector<Point2f> corners, Scalar color);
void draw_affine(Mat &img, Point2f center, int half_width, Mat A, Scalar color);
void draw_perspective(Mat &img, Point2f p, int half_width, Mat H, Scalar color);
void draw_transform(Mat &img, Point2f p, int half_width, Mat T, Scalar color);
void draw_dashed_line(Mat &img, Point2f p0, Point2f p1, Scalar color);