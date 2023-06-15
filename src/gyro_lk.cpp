#include "gyro_lk.hpp"
#include "utils.hpp"
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <filesystem>
#include <sstream>

#define VISUALIZE_LK 0

namespace fs = std::filesystem;
using namespace cv;
using namespace std;

static inline double det33d(const Matx33d H){
    return (
        H(0, 0) * H(1, 1) * H(2, 2) - H(0, 0) * H(1, 2) * H(2, 1)
      - H(0, 1) * H(1, 0) * H(2, 2) + H(0, 1) * H(1, 2) * H(2, 0)
      + H(0, 2) * H(1, 0) * H(2, 1) - H(0, 2) * H(1, 1) * H(2, 0)
    );
}

static inline Matx33d inv33d(const Matx33d H){
    double det = det33d(H);
    
    if (det == 0) {
        throw std::invalid_argument("The matrix is singular and cannot be inverted.");
    }

    Matx33d adj;  // Adjugate matrix

    adj(0, 0) = (H(1, 1) * H(2, 2) - H(2, 1) * H(1, 2));
    adj(0, 1) = (H(0, 2) * H(2, 1) - H(0, 1) * H(2, 2));
    adj(0, 2) = (H(0, 1) * H(1, 2) - H(0, 2) * H(1, 1));
    adj(1, 0) = (H(1, 2) * H(2, 0) - H(1, 0) * H(2, 2));
    adj(1, 1) = (H(0, 0) * H(2, 2) - H(0, 2) * H(2, 0));
    adj(1, 2) = (H(1, 0) * H(0, 2) - H(0, 0) * H(1, 2));
    adj(2, 0) = (H(1, 0) * H(2, 1) - H(2, 0) * H(1, 1));
    adj(2, 1) = (H(2, 0) * H(0, 1) - H(0, 0) * H(2, 1));
    adj(2, 2) = (H(0, 0) * H(1, 1) - H(1, 0) * H(0, 1));

    return adj * (1/det);
}

static inline Matx33d get_mat(const vector<Point2f> p0){
    auto M = 
        Matx33d(p0[0].x, p0[1].x, p0[2].x,
                p0[0].y, p0[1].y, p0[2].y,
                      1,       1,       1);

    auto b = Vec3d(p0[3].x, p0[3].y, 1);

    auto r = inv33d(M) * b;
    return {M(0,0)*r(0), M(0,1)*r(1), M(0,2)*r(2),
            M(1,0)*r(0), M(1,1)*r(1), M(1,2)*r(2),
            M(2,0)*r(0), M(2,1)*r(1), M(2,2)*r(2)};
}

static inline void gyro_lk_one_level(const GyroPredictType predict_type,
    const Mat_<uchar> &I0, const Mat_<uchar> &I1, const Mat_<Vec2s> &dI0, const vector<Point2f> &p0, vector<Point2f> &p1, 
    vector<uchar> &status, int level, int max_level, int half_width, float epsilon, int max_iterations,
    const vector<Matx33d> predicted_transform, vector<int> &iterations, const Range &range)
{
    const int hw{half_width};
    const int fw{2*half_width+1};

    // typedef Matx22f mat_type;
    // typedef Vec2f vec_type;

    // Buffers for interpolated values of the template patch and its derivatives
    cv::Mat_<float> T(fw, fw);
    // vector<vector<vec_type>> dT(fw, vector<vec_type>(fw));
    cv::Mat_<float> Tx(fw, fw);
    cv::Mat_<float> Ty(fw, fw);

    for (int i = range.start; i < range.end; i++){
        if (!status[i]) continue;

        cv::Point2f x0 = p0[i]*(1.f/(1 << level));
        if( level == max_level ) {
            if (predict_type != GyroPredictType::NONE){
                p1[i] = p1[i]*(1.f/(1 << level));
            } else {
                p1[i] = x0;
            }
        } else {
            p1[i] = p1[i] * 2.f;
        }

        const Point2f hw2(hw, hw);
        x0 -= hw2;
        Point2f x1 = p1[i] - hw2;
        Point2f ix1{std::floor(x1.x), std::floor(x1.y)};
        Point2f ix0{std::floor(x0.x), std::floor(x0.y)};

        if (ix0.x < -fw || ix0.x >= I0.cols || ix0.y < -fw || ix0.y >= I0.rows) {
            if (level == 0)
                status[i] = false;
            p1[i] = x1 + hw2;
            continue;
        }

        Matx33d P;
        vector<Point2f> perspective_patch;
        double det_P_inv;
        if (predict_type == GyroPredictType::PERSPECTIVE){
            P = predicted_transform[i];
            det_P_inv = 1.0 / det33d(P);
            perspective_patch = vector<Point2f>(fw * fw);
        }

        // Values used in bilinear interpolation of template image
        float rx = x0.x - ix0.x, ry = x0.y - ix0.y;
        float a = 1.0f - rx, b = 1.0f - ry;
        float w00 = b * a;
        float w01 = b * rx;
        float w10 = ry * a;
        float w11 = ry * rx;

        float H11{}, H12{}, H22{};
        // mat_type H = mat_type::zeros();
        for (int j = 0; j < fw; j++) {
            for (int k = 0; k < fw; k++) {
                int x = ix0.x + k;
                int y = ix0.y + j;

                Point3f p;
                if (predict_type == GyroPredictType::PERSPECTIVE){
                    p.x = P(0, 0) * (j - hw) + P(0, 1) * (k - hw) + P(0, 2);
                    p.y = P(1, 0) * (j - hw) + P(1, 1) * (k - hw) + P(1, 2);
                    p.z = P(2, 0) * (j - hw) + P(2, 1) * (k - hw) + P(2, 2);
                    perspective_patch[j * fw + k] = {p.x / p.z, p.y / p.z};
                }

                float tx, ty, t;
                float dTx, dTy;
                // Interpolate and store template values for subpixel reference points
                dTx = w00 * dI0(y, x)[0] + w01 * dI0(y, x + 1)[0] + w10 * dI0(y + 1, x)[0] + w11 * dI0(y + 1, x + 1)[0];
                dTy = w00 * dI0(y, x)[1] + w01 * dI0(y, x + 1)[1] + w10 * dI0(y + 1, x)[1] + w11 * dI0(y + 1, x + 1)[1];
                t = w00 * I0(y, x) + w01 * I0(y, x + 1) + w10 * I0(y + 1, x) + w11 * I0(y + 1, x + 1);

                if (predict_type == GyroPredictType::PERSPECTIVE){
                    Matx22d dW_dx_inv = p.z * det_P_inv *
                        Matx22d(-P(2,1)*p.y + P(1,1)*p.z,  P(2,1)*p.x - P(0,1)*p.z,
                                 P(2,0)*p.y - P(1,0)*p.z, -P(2,0)*p.x + P(0,0)*p.z);

                    tx = dTx * dW_dx_inv(0,0) + dTy * dW_dx_inv(1,0);
                    ty = dTx * dW_dx_inv(0,1) + dTy * dW_dx_inv(1,1); 
                } else {
                    tx = dTx;
                    ty = dTy;
                }

                H11 += tx * tx;
                H12 += tx * ty;
                H22 += ty * ty;
                // auto dt = vec_type(tx, ty);
                // dT[j][k] = dt;
                // H += dt * dt.t();

                T(j,k) = t;
                Tx(j,k) = tx;
                Ty(j,k) = ty;
            }
        }

        float D = H11*H22 - H12*H12;
        // float min_eig = (H22 + H11 - std::sqrt((H11-H22)*(H11-H22) +
                        // 4.f*H12*H12))/(2*fw*fw);

        // if (level == 0)
        //     printf("%f\n", min_eig);

        // if(D < FLT_EPSILON || min_eig < 5000) {
        if(D < FLT_EPSILON) {
            if (level == 0)
                status[i] = false;
            p1[i] = x1 + hw2;
            continue;
        }
        D = 1.f/D;

        Point2f p(0, 0);
        Point2f prev_delta_p;

#if VISUALIZE_LK
        cv::Mat_<Point2f> warp_idxs{fw, fw};
#endif

        int iter = 0;
        while (true)
        {
            Point2f ix1_next = ix1 + p;
            if( ix1_next.x < -fw || ix1_next.x >= I1.cols || ix1_next.y < -fw || ix1_next.y >= I1.rows )
            {
                if (level == 0)
                    status[i] = false;
                p1[i] = x1 + p + hw2;
                break;
            }

            // Values used in bilinear interpolation of new image
            float xp = x1.x + p.x, yp = x1.y + p.y;
            int ixp{0}, iyp{0};
            if (predict_type != GyroPredictType::PERSPECTIVE){
                ixp = std::floor(xp), iyp = std::floor(yp);
                rx = xp - ixp, ry = yp - iyp;
                a = 1.0f - rx, b = 1.0f - ry;
            }

            Point2f patch_center = x1 + p + hw2;
            double error = 0;
            float b1{}, b2{};
            // vec_type B{0, 0};
            for (int j = 0; j < fw; j++) {
                for (int k = 0; k < fw; k++) {
                    int x, y;
                    if (predict_type == GyroPredictType::PERSPECTIVE) {
                        Point2f patch_point = patch_center + perspective_patch[j * fw + k];
                        x = std::floor(patch_point.x);
                        y = std::floor(patch_point.y);
                        rx = patch_point.x - x; ry = patch_point.y - y;
                        a = 1.0f - rx; b = 1.0f - ry;
                    } else {
                        x = ixp + j;
                        y = iyp + k;
                    }

                    float I_val = b * (a * I1(y, x) + rx * I1(y, x + 1)) + ry * (a * I1(y + 1, x) + rx * I1(y + 1, x + 1));

                    float diff = I_val - T(k, j);
                    error += diff * diff;

                    b1 += Tx(k,j) * diff;
                    b2 += Ty(k,j) * diff;
                    // B += dT[khw][jhw] * diff;
#if VISUALIZE_LK
                    warp_idxs(k, j) = {(float)x, (float)y}; 
#endif
                }
            }

            // The image gradients have 5 more bits of integer precision 
            // See https://en.wikipedia.org/wiki/Fixed-point_arithmetic for more details
            b1 *= 32;
            b2 *= 32;
            // B[0] *= 32;
            // B[1] *= 32;

            Point2f delta_p{(H22*b1 - H12*b2) * D,
                            (H11*b2 - H12*b1) * D};
            // vec_type sol;
            // if (!solve(H, B, sol) || isnan(sol[0]))
                // throw std::runtime_error("Failed to solve for delta_p");
            // auto delta_p = Point2f{sol[0], sol[1]} ;
            p -= delta_p;

#if VISUALIZE_LK
            if (predict_type == GyroPredictType::PERSPECTIVE){
                Mat img1_out, img0_out, img_out;
                cv::cvtColor(I1, img1_out, cv::COLOR_GRAY2BGR);
                cv::cvtColor(I0, img0_out, cv::COLOR_GRAY2BGR);
                // auto c = x1 + p + hw2;
                draw_quad(img1_out, {warp_idxs(0,0), warp_idxs(0, fw-1), warp_idxs(fw-1,0), warp_idxs(fw-1,fw-1)}, COLOR_GREEN);
                draw_square(img0_out, x0 + hw2, hw, COLOR_CYAN);
                circle(img0_out, x0 + hw2, 2, COLOR_CYAN, -1);
                circle(img1_out, x1 + p + hw2, 2, COLOR_GREEN, -1);
                float scale = 1.0*(1 << level);
                resize(img0_out, img0_out, Size(), scale, scale, INTER_NEAREST);
                resize(img1_out, img1_out, Size(), scale, scale, INTER_NEAREST);
                hconcat(img0_out, img1_out, img_out);
                cv::imshow("KLT", img_out);
                int key = cv::waitKey(0);
                if (key == 'q') {
                    exit(0);
                } else if (key == 's'){
                    imwrite(SOURCE_DIR "viz_lk_d.png", img_out);
                    imwrite(SOURCE_DIR "viz_lk_s0.png", img0_out);
                    imwrite(SOURCE_DIR "viz_lk_s1.png", img1_out);
                }
            }
#endif

            if(delta_p.ddot(delta_p) <= epsilon*epsilon || iter >= max_iterations){
                p1[i] = x1 + p + hw2;
#if FILTER_ON_ERROR
                if (level == 0){
                    double error_norm = error/(hw*hw);
                    // cout << error_n rm << endl;
                    if (error_norm > 1000)
                        status[i] = false;
                }
#endif
                break;
            }

            // This is a hack to avoid getting stuck with slow convergence; taken from opencv
            if( iter > 0 && std::abs(delta_p.x + prev_delta_p.x) < 0.01 &&
               std::abs(delta_p.y + prev_delta_p.y) < 0.01 )
            {
                p1[i] = x1 + p + hw2 + delta_p*0.5f;
                break;
            }
            iter++;
            prev_delta_p = delta_p;
            
        }
        iterations[i] = iter;
    }
}

cv::Vec3d euler_angles_from_rot_mat(const cv::Matx33d& R)
{
    double sy = sqrt(R(0,0) * R(0,0) +  R(1,0) * R(1,0));

    bool singular = sy < 1e-6; // If

    double x, y, z;
    if (!singular)
    {
        x = atan2(R(2,1) , R(2,2));
        y = atan2(-R(2,0), sy);
        z = atan2(R(1,0), R(0,0));
    }
    else
    {
        x = atan2(-R(1,2), R(1,1));
        y = atan2(-R(2,0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}

GyroPredictType predict_type_from_string(const std::string &str){
    if (str == "NONE")
        return GyroPredictType::NONE;
    else if (str == "TRANSLATION")
        return GyroPredictType::TRANSLATION;
    else if (str == "PERSPECTIVE")
        return GyroPredictType::PERSPECTIVE;
    else if (str == "ADAPTIVE")
        return GyroPredictType::ADAPTIVE;
    else
        throw std::runtime_error("Unknown gyro prediction type: " + str);
}

std::string string_from_predict_type(GyroPredictType type){
    switch(type){
        case GyroPredictType::NONE:
            return "NONE";
        case GyroPredictType::TRANSLATION:
            return "TRANSLATION";
        case GyroPredictType::PERSPECTIVE:
            return "PERSPECTIVE";
        case GyroPredictType::ADAPTIVE:
            return "ADAPTIVE";
        default:
            throw std::runtime_error("NOT IMPLEMENTED");
    }
}

void gyro_aided_lk(LKState &state, LKInput &in){
    assert(in.img_pyr0.size() == (in.pyr_levels+1)*2);
    assert(in.img_pyr0.size() == in.img_pyr1.size());
    assert(state.pts0.size() == state.pts1.size());
    assert(in.img_pyr0[1].type() == CV_16SC2);
    assert(in.img_pyr0[0].type() == CV_8U);
    assert(in.img_pyr1[0].type() == CV_8U);
    //TODO: check that images have correct padding
    assert(state.status.size() == state.pts0.size());

    for (int level = in.pyr_levels; level >= 0; level--)
    {
#if VISUALIZE_LK
        Range range = Range(0, state.pts0.size());
#else
        parallel_for_(cv::Range(0, state.pts0.size()), [&](const cv::Range &range) {
#endif
        gyro_lk_one_level(state.predict_type, in.img_pyr0[level * 2], in.img_pyr1[level * 2], in.img_pyr0[level * 2 + 1], state.pts0, state.pts1,
               state.status, level, in.pyr_levels, in.half_patch_size, in.epsilon, in.max_iterations, state.predicted_transform,
               state.iterations, range);
#if !VISUALIZE_LK
        });
#endif
    }
}

Point2f distort_point(Point2f pt_norm_undist, const Vec4d &D, bool fisheye){
    float x = pt_norm_undist.x;
    float y = pt_norm_undist.y;
    if (fisheye) {
        float r = sqrt(x * x + y * y);
        float theta = atan(r);
        float theta2 = theta * theta;
        float theta4 = theta2 * theta2;
        float theta6 = theta4 * theta2;
        float theta8 = theta4 * theta4;
        float theta_d = theta * (1 + D(0) * theta2 + D(1) * theta4 + D(2) * theta6 + D(3) * theta8);
        float scaling = (r == 0) ? 1.0 : theta_d / r;
        return {x * scaling, y * scaling};
    } else {
        float r2 = x * x + y * y;
        float r4 = r2 * r2;
        // float r6 = r4 * r2;
        return {
            (float)(x * (1 + D(0) * r2 + D(1) * r4 /*+ D(4) * r6*/) + 2 * D(2) * x * y + D(3) * (r2 + 2 * x * x)),
            (float)(y * (1 + D(0) * r2 + D(1) * r4 /*+ D(4) * r6*/) + D(2) * (r2 + 2 * y * y) + 2 * D(3) * x * y)
        };
    }
}

void undistort_points(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &pts_undist,
                                    const Matx33d &K, const Vec4d &D, bool fisheye){
    if (fisheye) 
        cv::fisheye::undistortPoints(pts, pts_undist, K, D, cv::Mat(), K);
    else 
        cv::undistortPoints(pts, pts_undist, K, D, cv::Mat(), K);
}

void normalize_points(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &pts_undist,
                                    const Matx33d &K, const Vec4d &D, bool fisheye){
    if (fisheye) 
        cv::fisheye::undistortPoints(pts, pts_undist, K, D);
    else 
        cv::undistortPoints(pts, pts_undist, K, D);
}

void get_patch_indexes(vector<Point2f> &patch, Point2f p, int hw) {
    int fw = 2 * hw + 1;
    patch = vector<Point2f>(fw * fw);
    for (int i = 0; i < fw; i++) {
        for (int j = 0; j < fw; j++) {
            patch[i * fw + j] = {p.x + i - hw, p.y + j - hw};
        }
    }
}

static inline Point2f predict_point(const Point2f pt_norm, const Matx33d &R, const Matx33d &K, const Vec4d &D, bool use_fisheye) {
    float x_n = pt_norm.x;
    float y_n = pt_norm.y;
    float x = x_n * R(0,0) + y_n * R(0,1) + R(0,2);
    float y = x_n * R(1,0) + y_n * R(1,1) + R(1,2);
    float w = x_n * R(2,0) + y_n * R(2,1) + R(2,2);
    x /= w;
    y /= w;

    Point2f pt_norm_dist = distort_point({x, y}, D, use_fisheye);
    float u = K(0,0) * pt_norm_dist.x + K(0,2);
    float v = K(1,1) * pt_norm_dist.y + K(1,2);

    return {u, v};
}

void gyro_prediction(LKState &state, const GyroPredictInput &in){
    if (state.predict_type == GyroPredictType::NONE) return;
    else if (state.predict_type == GyroPredictType::ADAPTIVE){
        if (abs(atan2(in.R(1,0), in.R(0,0))) > in.adaptive_thresh)
            state.predict_type = GyroPredictType::PERSPECTIVE;
        else
            state.predict_type = GyroPredictType::TRANSLATION;
    }
    const float hw{(float)in.half_patch_size};

    Matx33d KRKinv;
    if (in.ignore_distortion)
        KRKinv = in.K * in.R * inv33d(in.K);

    if (state.predict_type == GyroPredictType::TRANSLATION){
        vector<Point2f> pts_norm(state.pts0.size());
        normalize_points(state.pts0, pts_norm, in.K, in.D, in.fisheye);
        for (size_t i = 0; i < state.pts0.size(); i++) {
            state.pts_pred[i] = predict_point(pts_norm[i], in.R, in.K, in.D, in.fisheye);
        }
        state.pts1 = state.pts_pred;
        return;
    }

    const vector<Point2f> patch_corners_with_center = {{0,0}, {-hw,-hw}, {hw,-hw}, {-hw,hw}, {hw,hw}};
    const vector<Point2f> patch_corners = {{-hw,-hw}, {hw,-hw}, {-hw,hw}, {hw,hw}};
    const Matx33d A_inv = inv33d(get_mat(patch_corners));

    vector<Point2f> corners_dist(state.pts0.size()*5);
    auto corners_norm = corners_dist;

    for (size_t i = 0; i < corners_dist.size(); i++) {
        corners_dist[i] = state.pts0[i/5] + patch_corners_with_center[i%5];
    }
    normalize_points(corners_dist, corners_norm, in.K, in.D, in.fisheye);

    for (size_t i = 0; i < corners_norm.size(); i++){
        int idx = i/5;
        int irem = i%5;
        Point2f pt_pred;
        if (in.ignore_distortion){
            Vec3f res = KRKinv * Vec3d{corners_dist[i].x, corners_dist[i].y, 1};
            pt_pred = {res[0]/res[2], res[1]/res[2]};
        }else
            pt_pred = predict_point(corners_norm[i], in.R, in.K, in.D, in.fisheye);

        if (irem == 0){
            state.pts_pred[idx] = pt_pred;
            if (pt_pred.x < 0 || pt_pred.x >= in.width || pt_pred.y < 0 || pt_pred.y >= in.height){
                state.status[idx] = 0;
                i += 4;
            }
        } else {
            state.patch_corners[idx][irem-1] = pt_pred - state.pts_pred[idx];
        }
        if (irem == 4){
            auto B = get_mat(state.patch_corners[idx]);
            state.predicted_transform[idx] = B * A_inv;
        }
    }
    state.pts1 = state.pts_pred;
}

static inline void integrate_one_imu_measurement(cv::Point3f &gyro, double dt, Matx33d &dR)
{
    static const Point3f bias{0};
    const float x = (gyro.x - bias.x) * dt;
    const float y = (gyro.y - bias.y) * dt;
    const float z = (gyro.z - bias.z) * dt;

    const float d2 = x * x + y * y + z * z;
    const float d = std::sqrt(d2);

    cv::Matx33d I = cv::Matx33d::eye();
    cv::Matx33d W{0, -z,  y,
                  z,  0, -x,
                 -y,  x,  0};

    if (d < 1e-4){
        dR = I + W; // on-manifold equation (4)
    }
    else {
        dR = I + W * (std::sin(d) / d) + W * W * ((1.0f - std::cos(d)) / d2); // on-manifold equation (3)
    }
}

void integrate_imu_measurements(double t, double t_ref, const std::vector<ImuMeas> imu_meas, const Matx33d &Rbc, Matx33d &R){
    cv::Matx33d dR_ref_cur = cv::Matx33d::eye();
    const int n = imu_meas.size()-1;

    cv::Point3f ang_vel_total;
    // Consider the gap between the IMU timestamp and camera timestamp.
    for (int i = 0; i < n; i++) {
        float tstep;
        cv::Point3f ang_vel;
        double dt = imu_meas[i+1].t - imu_meas[i].t;
        //FIXME: this should not be allowed to happen
        if (dt < DBL_EPSILON) continue;
        if((i == 0) && (i < (n-1)))
        {
            float tini = imu_meas[i].t - t_ref;
            ang_vel = (imu_meas[i].w + imu_meas[i+1].w -
                    (imu_meas[i+1].w - imu_meas[i].w) * (tini/dt)) * 0.5f;
            tstep = imu_meas[i+1].t - t_ref;
        }
        else if(i < (n-1))
        {
            ang_vel = (imu_meas[i].w + imu_meas[i+1].w) * 0.5f;
            tstep = dt;
        }
        else if((i > 0) && (i == (n-1)))
        {
            float tend = imu_meas[i+1].t - t;
            ang_vel = (imu_meas[i].w + imu_meas[i+1].w -
                    (imu_meas[i+1].w - imu_meas[i].w) * (tend / dt)) * 0.5f;
            tstep = t - imu_meas[i].t;
        }
        // else if((i == 0) && (i == (n-1)))
        else
        {
            ang_vel = imu_meas[i].w;
            tstep = t - t_ref;
        }
        Matx33d dR;
        ang_vel_total += ang_vel;
        integrate_one_imu_measurement(ang_vel, tstep, dR);
        dR_ref_cur = dR_ref_cur * dR;
    }

    R = Rbc.t() * dR_ref_cur.t() * Rbc;
};

void perform_ransac(LKState &state, const cv::Matx33d &K, const cv::Vec4d &D, double threshold, double confidence, bool use_fisheye)
{
    if (countNonZero(state.status) < 15){
        state.status = vector<uchar>();
        return;
    }
    vector<Point2f> pts0_rsc, pts1_rsc;
    for (size_t i = 0; i < state.pts0.size(); i++) {
        if (state.status[i]){
            pts0_rsc.push_back(state.pts0[i]);
            pts1_rsc.push_back(state.pts1[i]);
        }
    }

    // Need normalized image coordinates for fundamental matrix
    normalize_points(pts0_rsc, pts0_rsc, K, D, use_fisheye);
    normalize_points(pts1_rsc, pts1_rsc, K, D, use_fisheye);

    vector<uchar> mask_rsc;
    cv::findFundamentalMat(pts0_rsc, pts1_rsc, mask_rsc, cv::FM_RANSAC, threshold, confidence);
    // cv::findEssentialMat(pts0_rsc, pts1_rsc, Matx33d::eye(), cv::FM_RANSAC, confidence, threshold, mask_rsc);

    int offset = 0;
    for (size_t i = 0; i < state.pts0.size(); i++) {
        if (state.status[i])
            state.status[i] = mask_rsc[i - offset];
        else
            offset++;
    }
}

static inline vector<Point2f> get_absolute_corners(const vector<Point2f> &corners, Point2f center){
    vector<Point2f> abs_corners;
    for (const auto &c : corners)
        abs_corners.push_back(c + center);
    return abs_corners;
}

// FIXME: this is a bit dirty
Mat img_dist_out, img0_dist_out, img1_dist_out;

void draw_dist_and_undist_predictions(const cv::Mat &img0_bgr, const cv::Mat &img1_bgr, const LKState &state, const cv::Matx33d &R,
                                      const cv::Matx33d &K, const cv::Vec4d &D, int hw, bool use_fisheye) {
    cv::Mat img0_undist, img1_undist, img0_out, img1_out;
    img0_bgr.copyTo(img0_out);
    img1_bgr.copyTo(img1_out);
    if (use_fisheye) {
        cv::fisheye::undistortImage(img0_bgr, img0_undist, K, D, K);
        cv::fisheye::undistortImage(img1_bgr, img1_undist, K, D, K);
    }
    else {
        cv::undistort(img0_bgr, img0_undist, K, D, K);
        cv::undistort(img1_bgr, img1_undist, K, D, K);
    }

    auto &p0 = state.pts0;
    vector<Point2f> corners_dist(p0.size()*4), corners_undist, pts_pred_undist;
    vector<Point2f> corners0_dist(p0.size()*4), corners0_undist, pts0_undist;
    vector<Point> corner_modifiers = {{-hw, -hw}, {-hw, hw}, {hw, -hw}, {hw, hw}};

    for (size_t i = 0; i < corners_dist.size(); i++) {
        corners_dist[i] = state.pts_pred[i/4] + state.patch_corners[i/4][i%4];
        corners0_dist[i] = p0[i/4] + (Point2f)corner_modifiers[i%4];
    }

    undistort_points(corners_dist, corners_undist, K, D, use_fisheye);
    undistort_points(state.pts_pred, pts_pred_undist, K, D, use_fisheye);
    undistort_points(corners0_dist, corners0_undist, K, D, use_fisheye);
    undistort_points(p0, pts0_undist, K, D, use_fisheye);

    Scalar orig_color = COLOR_CYAN;
    Scalar pred_color = COLOR_YELLOW;

    for (int idx = 0; idx < p0.size(); idx++){
        vector<Point2f> c0_dist = vector<Point2f>(corners0_dist.begin()+idx*4, corners0_dist.begin()+(idx+1)*4);
        vector<Point2f> c0_undist = vector<Point2f>(corners0_undist.begin()+idx*4, corners0_undist.begin()+(idx+1)*4);
        draw_quad(img0_out, c0_dist, orig_color);
        draw_quad(img0_undist, c0_undist, orig_color);
        circle(img0_out, p0[idx], 2, orig_color, -1);
        circle(img0_undist, pts0_undist[idx], 2, orig_color, -1);

        auto &pp = state.pts_pred[idx];
        vector<Point2f> c_dist = vector<Point2f>(corners_dist.begin()+idx*4, corners_dist.begin()+(idx+1)*4);
        vector<Point2f> c_undist = vector<Point2f>(corners_undist.begin()+idx*4, corners_undist.begin()+(idx+1)*4);
        if (pp.x < -hw || pp.x > img0_out.cols + hw || pp.y < -hw || pp.y > img0_out.rows + hw)
            continue;
        draw_quad(img1_out, c_dist, pred_color);
        draw_quad(img1_undist, c_undist, pred_color);
        circle(img1_out, state.pts_pred[idx], 2, pred_color, -1);
        circle(img1_undist, pts_pred_undist[idx], 2, pred_color, -1);
    }

    // Mat img0_dist_out, img1_dist_out;
    hconcat(img0_out, img0_undist, img0_dist_out);
    hconcat(img1_out, img1_undist, img1_dist_out);
    vconcat(img0_dist_out, img1_dist_out, img_dist_out);
    imshow("distorted vs undistorted", img_dist_out);
}

void draw_lines(const cv::Mat &img0_bgr, const cv::Mat &img1_bgr, const LKState &state)
{
    Mat img1_out = img1_bgr.clone();
    Mat img0_out = img0_bgr.clone();
    int w = img0_bgr.cols;
    
    Mat img_out;
    hconcat(img0_out, img1_out, img_out);

    for (size_t i = 0; i < state.pts0.size(); i++){
        if (!state.status[i]) continue;
        line(img_out, state.pts0[i], state.pts1[i] + Point2f(w, 0), COLOR_YELLOW, 1, LINE_AA);
    }
    resize(img_out, img_out, Size(), 1.5, 1.5);
    imshow("lines", img_out);
}

void draw_results(const cv::Mat &img0_bgr, const cv::Mat &img1_bgr, const LKState &state, const LKState &comp, int hw,
                  double relative_timestamp, std::string save_folder) {
    static bool run_to_timestamp{false};
    static bool step_mode{false};
    static double target_timestamp{0};

    if (run_to_timestamp && relative_timestamp >= target_timestamp){
        run_to_timestamp = false;
        step_mode = true;
    }

    Mat img1_out = img1_bgr.clone();
    Mat img0_out = img0_bgr.clone();
    for (size_t i = 0; i < state.pts0.size(); i++)
    {
        // if (!gyro_output.status[i]) continue;
        draw_square(img0_out, state.pts0[i], hw, COLOR_CYAN);
        circle(img0_out, state.pts0[i], 2, COLOR_CYAN, -1);

        if (state.status[i]){
            Scalar draw_color = COLOR_GREEN;
            if (comp.status[i])
                draw_color = COLOR_BLUE;
            else
                draw_color = COLOR_GREEN;
            if (state.predict_type != GyroPredictType::NONE)
                draw_quad(img1_out, get_absolute_corners(state.patch_corners[i], state.pts1[i]), draw_color);
            line(img1_out, state.pts0[i], state.pts1[i], draw_color, 1, LINE_AA);
            circle(img1_out, state.pts1[i], 2, draw_color, -1);        
        }

        if (state.predict_type != GyroPredictType::NONE){
            Scalar pred_color = COLOR_YELLOW;
            if (!state.status[i] && !comp.status[i])
                pred_color = COLOR_RED;
            
            auto &pp = state.pts_pred[i];
            if (!(pp.x < -hw || pp.x > img0_out.cols + hw || pp.y < -hw || pp.y > img0_out.rows + hw)){
                circle(img1_out, pp, 2, pred_color, -1);        
                line(img1_out, state.pts0[i], pp, pred_color, 1, LINE_AA);
            }
        }

        if (comp.status[i] && !state.status[i]){
            Scalar comp_color = COLOR_CYAN;
            line(img1_out, comp.pts0[i], comp.pts1[i], comp_color, 1, LINE_AA);
            circle(img1_out, comp.pts1[i], 2, comp_color, -1);        
        }
    }

    Mat img_out, img_out_resize;
    hconcat(img0_out, img1_out, img_out);

    float scale = 1.5;
    resize(img_out, img_out_resize, Size(), scale, scale);
    imshow("results", img_out_resize);

    string lk_name = toLowerCase(string_from_predict_type(state.predict_type)).substr(0, 3);
    string comp_name = toLowerCase(string_from_predict_type(comp.predict_type)).substr(0, 3);
    ostringstream t_ss;
    t_ss << std::fixed << std::setprecision(2) << relative_timestamp;
    string file_prefix = "/" + lk_name + "_" + comp_name + "_" + t_ss.str();
    string file_dist = "/dist_" + lk_name + "_" + t_ss.str() + ".png";
    string file_dist0 = "/dist0_" + lk_name + "_" + t_ss.str() + ".png";
    string file_dist1 = "/dist1_" + lk_name + "_" + t_ss.str() + ".png";

    if (step_mode) {
        for (;;) {
            int key = cv::waitKey(0);
            if (key == 'q')
                exit(0);
            else if (key == 'n') {
                step_mode = false;
                break;
            } else if (key == 's') {
                std::string file_single0 = save_folder + file_prefix + "s0" + ".png";
                std::string file_single1 = save_folder + file_prefix + "s1" + ".png";
                std::string file_double = save_folder + file_prefix + "d" + ".png";
                cout << "\nSaving to " << file_single1 << endl;
                cv::imwrite(file_single1, img1_out);
                cv::imwrite(file_single0, img0_out);
                cv::imwrite(file_double, img_out);
                if (!img_dist_out.empty()){
                    cv::imwrite(save_folder + file_dist, img_dist_out);
                    cv::imwrite(save_folder + file_dist0, img0_dist_out);
                    cv::imwrite(save_folder + file_dist1, img1_dist_out);
                }
                break;
            } else if (key == ' ')
                break;
            else if (key == 't'){
                cout << "\nRun to timestamp: " << flush;
                cin >> target_timestamp;
                run_to_timestamp = true;
                step_mode = false;
                break;
            }
        }
    } else {
        int key = cv::waitKey(1);
        if (key == 'n')
            step_mode = true;
        else if (key == 'q')
            exit(0);
    }
}