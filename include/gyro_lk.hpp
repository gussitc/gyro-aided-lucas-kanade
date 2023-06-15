#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>

class ImuMeas
{
public:
    ImuMeas(){}
    ImuMeas(const float &acc_x, const float &acc_y, const float &acc_z,
             const float &ang_vel_x, const float &ang_vel_y, const float &ang_vel_z,
             const double &timestamp): a(acc_x,acc_y,acc_z), w(ang_vel_x,ang_vel_y,ang_vel_z), t(timestamp){}
    ImuMeas(const cv::Point3f acc, const cv::Point3f gyro, const double &timestamp):
        a(acc.x,acc.y,acc.z), w(gyro.x,gyro.y,gyro.z), t(timestamp){}

public:
    cv::Point3f a;
    cv::Point3f w;
    double t;
};

struct GyroPredictInput{
    const cv::Matx33d &R;
    const cv::Matx33d &K;
    const cv::Vec4d &D;
    int width;
    int height;
    int half_patch_size;
    bool fisheye = false;
    double adaptive_thresh = 0;
    bool ignore_distortion = false;
};

enum class GyroPredictType {
    NONE,
    TRANSLATION,
    PERSPECTIVE,
    ADAPTIVE
};

struct LKInput{
    const std::vector<cv::Mat> &img_pyr0;
    const std::vector<cv::Mat> &img_pyr1;
    int pyr_levels;
    int half_patch_size;
    double epsilon;
    int max_iterations;
};

struct LKState {
    GyroPredictType predict_type;
    std::vector<cv::Point2f> pts0;
    std::vector<cv::Point2f> pts1;
    std::vector<cv::Point2f> pts_pred;
    std::vector<uchar> status;
    // std::vector<uchar> status_pred;
    std::vector<float> predict_error;
    std::vector<std::vector<cv::Point2f>> patch_corners;
    std::vector<int> iterations;
    std::vector<cv::Matx33d> predicted_transform;

    LKState(GyroPredictType predict_type, std::vector<cv::Point2f> pts0) : predict_type{predict_type}, pts0{pts0} {
        reset();
    }

    void reset() {
        pts1 = pts_pred = pts0;
        status = std::vector<uchar>(pts0.size(), 1);
        predict_error = std::vector<float>(pts0.size(), 0.0);
        patch_corners = std::vector<std::vector<cv::Point2f>>(pts0.size(), std::vector<cv::Point2f>(4));
        iterations = std::vector<int>(pts0.size(), 0);
        predicted_transform = std::vector<cv::Matx33d>(pts0.size(), cv::Matx33d::eye());
    }

    void update() {
        pts0.clear();
        for(int i = 0; i < status.size(); i++){
            if (status[i])
                pts0.push_back(pts1[i]);
        }
        reset();
    }
};

cv::Vec3d euler_angles_from_rot_mat(const cv::Matx33d& R);
GyroPredictType predict_type_from_string(const std::string &str);
std::string string_from_predict_type(GyroPredictType type);

cv::Point2f distort_point(cv::Point2f pt_norm_undist, const cv::Vec4d &D, bool fisheye);
void gyro_aided_lk(LKState &state, LKInput &input);

void undistort_points(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &pts_undist, const cv::Matx33d &K, const cv::Vec4d &D,
                      bool fisheye = false);

void normalize_points(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &pts_undist, const cv::Matx33d &K, const cv::Vec4d &D,
                      bool fisheye = false);

void gyro_prediction(LKState &state, const GyroPredictInput &in);
void get_patch_indexes(std::vector<cv::Point2f> &patch, cv::Point2f p, int hw);
void get_patch_corners(std::vector<cv::Point2f> &patch_map, cv::Point2f p, const cv::Matx33d &R, const cv::Matx33d &K,
                      const cv::Vec4d &D, int hw, bool use_fisheye);

void integrate_imu_measurements(double t, double t_ref, const std::vector<ImuMeas> imu_meas, const cv::Matx33d &Rbc, cv::Matx33d &R);

void perform_ransac(LKState &state, const cv::Matx33d &K, const cv::Vec4d &D, double threshold = 3.0, double confidence = 0.99,
                    bool use_fisheye = false);

void draw_dist_and_undist_predictions(const cv::Mat &img0_bgr, const cv::Mat &img1_bgr, const LKState &state, const cv::Matx33d &R,
                                      const cv::Matx33d &K, const cv::Vec4d &D, int half_patch_size, bool use_fisheye);

void draw_results(const cv::Mat &img0_bgr, const cv::Mat &img1_bgr, const LKState &state, const LKState &comp, int half_patch_size,
                  double relative_timestamp, std::string save_folder = "");

void draw_lines(const cv::Mat &img0_bgr, const cv::Mat &img1_bgr, const LKState &state);