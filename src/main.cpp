#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "gyro_lk.hpp"
#include "load_data.hpp"
#include <filesystem>

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

static const string output_base_path(SOURCE_DIR "results");
static const string figure_path(SOURCE_DIR "figures");

using namespace std;
using namespace cv;
namespace fs = filesystem;

int main(int, char**) {
    int half_patch_size{0}, pyr_levels{-1}, max_iterations{0}, max_features{-1}, visualize{-1},
        get_new_keypoints{-1}, feature_min_dist{0}, write_to_file{-1}, replace_results{-1}, undistort_images{-1}, start_idx{-1};
    double epsilon{0}, feature_quality{0}, ransac_thresh{0}, adaptive_thresh{-1};
    LOAD_CONFIG(half_patch_size);
    LOAD_CONFIG(pyr_levels);
    LOAD_CONFIG(epsilon);
    LOAD_CONFIG(max_iterations);
    LOAD_CONFIG(max_features);
    LOAD_CONFIG(visualize);
    LOAD_CONFIG(get_new_keypoints);
    LOAD_CONFIG(feature_quality);
    LOAD_CONFIG(feature_min_dist);
    LOAD_CONFIG(write_to_file);
    LOAD_CONFIG(replace_results);
    LOAD_CONFIG(undistort_images);
    LOAD_CONFIG(start_idx);
    LOAD_CONFIG(ransac_thresh);
    LOAD_CONFIG(adaptive_thresh);

    std::vector<std::vector<ImuMeas>> imu_meas_vec;
	std::vector<cv::Mat> images;
	std::vector<double> image_timestamps;
    get_data_from_rosbag(imu_meas_vec, images, image_timestamps);
    assert(images.size() == imu_meas_vec.size() + 1);

    Matx33d K, Rbc, K_og;
    Vec4d D, D_og;
    bool use_fisheye, use_fisheye_og;
    vector<int> resolution;
    load_camera_params(K, D, Rbc, use_fisheye, resolution);
    K_og = K;
    D_og = D;
    use_fisheye_og = use_fisheye;
    if (undistort_images){
        D = cv::Vec4d::all(0);
        use_fisheye = false;
    }

    ofstream angvel_fs, pred_fs, lk_fs, match_fs, points_fs, error_fs;
    string output_path;
    if (write_to_file) {
        if (replace_results){
            output_path = output_base_path + "_test/";
            fs::remove_all(output_path);
            fs::create_directory(output_path);
        } else {
            for (int j = 0;;j++) {
                output_path = output_base_path + to_string(j) + "/";
                if (fs::create_directory(output_path)) 
                    break;
            }
        }
        angvel_fs.open(output_path + "angular_velocity.txt", ios::app);
        pred_fs.open(output_path + "predict_runtimes.txt", ios::app);
        lk_fs.open(output_path + "lk_runtimes.txt", ios::app);
        match_fs.open(output_path + "matches.txt", ios::app);
        points_fs.open(output_path + "points.txt", ios::app);
        error_fs.open(output_path + "error.txt", ios::app);
    }

    vector<Point2f> p0;
    if (max_features > 0)
        goodFeaturesToTrack(images[start_idx], p0, max_features, feature_quality, feature_min_dist, Mat(), 2*half_patch_size + 1);
    vector<LKState> lk_states = {
        {GyroPredictType::NONE, p0},
        {GyroPredictType::TRANSLATION, p0},
        {GyroPredictType::PERSPECTIVE, p0},
        {GyroPredictType::ADAPTIVE, p0}
    };
    int lk_show_idx = 2;
    int lk_compare_idx = 0;
    LKState &adaptive_lk = lk_states[3];
    // string lk_type = "", lk_compare = "";
    // LOAD_CONFIG(lk_type);
    // LOAD_CONFIG(lk_compare); 
    // int lk_show_idx = (int)predict_type_from_string(lk_type);
    // int lk_compare_idx = (int)predict_type_from_string(lk_compare);

    // vector<LKState> lk_states = {{GyroPredictType::EUCLIDEAN, p0}};
    // LKState &euclid_lk = lk_states[0];

    for (int k = start_idx; k < imu_meas_vec.size(); k++) {
        // cout << "Progress: " << setprecision(0) << fixed << 100*(k+1)/(double)imu_meas_vec.size() << "%\r" << flush;
        cout << "Progress: " << k << " / " << imu_meas_vec.size() - 1 << "\r" << flush;
        adaptive_lk.predict_type = GyroPredictType::ADAPTIVE;

        Mat img0, img1, img0_bgr, img1_bgr, img0_undist, img1_undist;

        if (undistort_images){
            img0 = images[k].clone(); 
            img1 = images[k+1].clone();
            if (use_fisheye_og){
                cv::fisheye::undistortImage(img0, img0, K_og, D_og, K_og);
                cv::fisheye::undistortImage(img1, img1, K_og, D_og, K_og);
            }
            else {
                cv::undistort(img0, img0, K_og, D_og, K_og);
                cv::undistort(img1, img1, K_og, D_og, K_og);
            }
        }
        else {
            img0 = images[k];
            img1 = images[k+1];
        }

        cvtColor(img0, img0_bgr, COLOR_GRAY2BGR); 
        cvtColor(img1, img1_bgr, COLOR_GRAY2BGR); 

        double timestamp_last = image_timestamps[k];
        double timestamp = image_timestamps[k+1];
        double t_out = timestamp - image_timestamps[0];
        vector<ImuMeas> imu_meas = imu_meas_vec[k];
        // Point3f w_total;

        Matx33d R;
        integrate_imu_measurements(timestamp, timestamp_last, imu_meas, Rbc, R);
        auto ang_vel = euler_angles_from_rot_mat(R);
        angvel_fs << t_out << " " << ang_vel[0] << " " << ang_vel[1] << " " << ang_vel[2] << " " << endl;

        bool ignore_distortion = false;
        GyroPredictInput predict_input{
            R, K, D, img0.cols, img0.rows, half_patch_size, (bool)use_fisheye, adaptive_thresh, ignore_distortion};
        pred_fs << t_out << " ";
        for (auto &state : lk_states) {
            int runtime = runtime_us([&]() {gyro_prediction(state, predict_input);});
            pred_fs << runtime << " ";
        }
        pred_fs << endl;

        vector<Mat> imgpyr0, imgpyr1;
        Size win_size(2*half_patch_size + 1, 2*half_patch_size + 1);
        cv::buildOpticalFlowPyramid(img0, imgpyr0, win_size, pyr_levels);
        cv::buildOpticalFlowPyramid(img1, imgpyr1, win_size, pyr_levels);

        LKInput lk_input{imgpyr0, imgpyr1, pyr_levels, half_patch_size, epsilon, max_iterations};

        lk_fs << t_out << " ";
        for (auto &state : lk_states) {
            int runtime = runtime_us([&](){gyro_aided_lk(state, lk_input);});
            lk_fs << runtime << " ";
        }
        lk_fs << endl;

        double max_focallength = std::max(K(0, 0), K(1, 1));
        double threshold = ransac_thresh/max_focallength;
        double confidence = 0.999;
        for (auto &state : lk_states) {
            if (countNonZero(state.status) < 15){
                // cout << "Not enough matches for RANSAC" << endl;
                // lkp.status = vector<uchar>(lkp.status.size(), 0);
                continue;
            }
            perform_ransac(state, K, D, threshold, confidence, use_fisheye);
        }

        match_fs  << t_out << " ";
        points_fs << t_out << " ";
        error_fs  << t_out << " ";
        for (auto &state : lk_states) {
            float track_rate = (float)countNonZero(state.status) / state.pts0.size();
            match_fs  << track_rate << " ";
            points_fs << countNonZero(state.status) << " ";

            if (state.predict_type == GyroPredictType::NONE){
                error_fs << "0 ";
                continue;
            }
            for (int i = 0; i < state.pts0.size(); i++){
                if (state.status[i] == 0)
                    continue;
                state.predict_error[i] = norm(state.pts_pred[i] - state.pts1[i]);
            }
            error_fs << mean(state.predict_error, state.status)[0] << " ";
        }
        match_fs  << endl;
        points_fs << endl;
        error_fs  << endl;

        if (visualize){
            // draw_lines(img0_bgr, img1_bgr, lk_states.at(lk_compare_idx));
            draw_dist_and_undist_predictions(img0_bgr, img1_bgr, lk_states.at(lk_show_idx), R, K, D, half_patch_size, use_fisheye);
            draw_results(img0_bgr, img1_bgr, lk_states.at(lk_show_idx), lk_states.at(lk_compare_idx), half_patch_size, t_out, figure_path);
        }

        if (get_new_keypoints && max_features > 0)
            goodFeaturesToTrack(images[k+1], p0, max_features, feature_quality, feature_min_dist, Mat(), 2*half_patch_size + 1);

        for (auto &state : lk_states){
            if (get_new_keypoints){
                state.pts0 = p0;
                state.reset();
                continue;
            }
            state.update();
        }
    }
    cout << endl << "Done" << endl;
    return 0;
}
