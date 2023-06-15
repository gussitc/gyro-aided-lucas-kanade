#include "load_data.hpp"
#include "utils.hpp"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
#include <string.h>
#include <random>

using namespace std;
using namespace cv;

void get_data_from_rosbag(std::vector<std::vector<ImuMeas>> &imu_meas, std::vector<cv::Mat> &images, std::vector<double> &image_timestamps)
{
    string bag_file{""};
    string image_topic{""};
    string imu_topic{""};
    int num_skip_frames{0};
    int normalize_images{-1};
    double timeoffset_std{-1};

    LOAD_CONFIG(bag_file);
    LOAD_CONFIG(image_topic);
    LOAD_CONFIG(imu_topic);
    LOAD_CONFIG(num_skip_frames);
    LOAD_CONFIG(normalize_images);
    LOAD_CONFIG(timeoffset_std);

    // Create a random number generator
    // std::random_device rd;
    std::mt19937 gen(0);
    // Create a normal distribution with mean and standard deviation
    std::normal_distribution<> dist(0.0, timeoffset_std);

    rosbag::Bag bag;
    bag.open(bag_file, rosbag::bagmode::Read);

    rosbag::View view(bag);
    int frame = 0;
    for (rosbag::MessageInstance const& m : view) {
        if (m.getTopic() == image_topic) {
            sensor_msgs::Image::ConstPtr image_msg = m.instantiate<sensor_msgs::Image>();
            if (image_msg != nullptr && frame++ % num_skip_frames == 0) {
                cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                if (normalize_images){
                    Mat dst;
                    equalizeHist( cv_ptr->image, dst );
                    images.push_back(dst);
                }
                else
                    images.push_back(cv_ptr->image);
                image_timestamps.push_back(image_msg->header.stamp.toSec());
                imu_meas.push_back(vector<ImuMeas>{});
            }
        }
    }
    imu_meas.pop_back(); // the last one will be empty

    int cur_image_idx = 1;

    for (rosbag::MessageInstance const& m : view) {
        if (m.getTopic() == imu_topic) {
            sensor_msgs::Imu::ConstPtr imu_msg = m.instantiate<sensor_msgs::Imu>();
            if (imu_msg != nullptr) {
                // Add imu_msg to imu_messages vector
                // imu_messages.push_back(*imu_msg);
                ImuMeas meas;
                double offset_s = dist(gen);
                meas.t = imu_msg->header.stamp.toSec() + offset_s;
                if (meas.t < image_timestamps[0])
                    continue;
                meas.w = Point3f(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                meas.a = Point3f(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                if (meas.t >= image_timestamps[cur_image_idx]){
                    cur_image_idx++;
                    if (cur_image_idx == images.size())
                        break;
                }
                imu_meas.at(cur_image_idx - 1).push_back(meas);
            }
        }
    }

    bag.close();
}

void load_camera_params(Matx33d &K, Vec4d &D, Matx33d &Rbc, bool &use_fisheye, vector<int> &resolution){
    string camchain_file{""};
    LOAD_CONFIG(camchain_file);
    FileStorage camchain(camchain_file, FileStorage::READ);

    vector<double> intrinsics;
    vector<double> distortion_coeffs;
    string distortion_model;

    vector<vector<double>> T_cam_imu, T_imu_cam;
    camchain["cam0"]["intrinsics"] >> intrinsics;
    camchain["cam0"]["distortion_coeffs"] >> D;
    camchain["cam0"]["T_cam_imu"] >> T_cam_imu;
    camchain["cam0"]["T_imu_cam"] >> T_imu_cam;
    camchain["cam0"]["distortion_model"] >> distortion_model;
    camchain["cam0"]["resolution"] >> resolution;

    if (distortion_model == "equidistant")
        use_fisheye = true;
    else if (distortion_model == "radtan")
        use_fisheye = false;
    else
        throw std::runtime_error("Unknown distortion model: " + distortion_model);

    if (!T_cam_imu.empty()){
        Rbc << T_cam_imu[0][0], T_cam_imu[0][1], T_cam_imu[0][2],
               T_cam_imu[1][0], T_cam_imu[1][1], T_cam_imu[1][2],
               T_cam_imu[2][0], T_cam_imu[2][1], T_cam_imu[2][2];
        Rbc = Rbc.t();
    }
    else if (!T_imu_cam.empty()) {
        Rbc << T_imu_cam[0][0], T_imu_cam[0][1], T_imu_cam[0][2],
               T_imu_cam[1][0], T_imu_cam[1][1], T_imu_cam[1][2],
               T_imu_cam[2][0], T_imu_cam[2][1], T_imu_cam[2][2];
    } else
        throw std::runtime_error("No camera imu transform found in camchain file");

    K = Matx33d(intrinsics[0], 0, intrinsics[2], 0, intrinsics[1], intrinsics[3], 0, 0, 1);
}