#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "gyro_lk.hpp"

void get_data_from_rosbag(std::vector<std::vector<ImuMeas>> &imu_meas, std::vector<cv::Mat> &images, std::vector<double> &image_timestamps);

void load_camera_params(cv::Matx33d &K, cv::Vec4d &D, cv::Matx33d &Rbc, bool &use_fisheye, std::vector<int> &resolution);