#include "utils.hpp"

static inline vector<Point2f> get_corners(Point2i center, int half_width){
    Point2i pt_tl = center - Point2i(half_width, half_width);
    Point2i pt_tr = center + Point2i(half_width, -half_width);
    Point2i pt_bl = center + Point2i(-half_width, half_width);
    Point2i pt_br = center + Point2i(half_width, half_width);
    return {pt_tl, pt_tr, pt_bl, pt_br};
}

void draw_quad(Mat &img, std::vector<Point2f> corners, Scalar color){
    line(img, corners[0], corners[1], color, line_thickness, LINE_AA);
    line(img, corners[1], corners[3], color, line_thickness, LINE_AA);
    line(img, corners[3], corners[2], color, line_thickness, LINE_AA);
    line(img, corners[2], corners[0], color, line_thickness, LINE_AA);
}

void draw_square(Mat &img, Point2f center, int half_width, Scalar color){
    auto corners = get_corners(center, half_width);
    draw_quad(img, corners, color);
}

void draw_affine(Mat &img, Point2f p, int half_width, Mat A, Scalar color){
    const float hw{(float)half_width};
    const vector<Point2f> patch_corners = {{-hw,-hw}, {hw,-hw}, {-hw,hw}, {hw,hw}};
    vector<Point2f> affine_corners(4);
    for (size_t j = 0; j < 4; j++){
        auto c = patch_corners[j];
        affine_corners[j].x = p.x + c.x * A.at<float>(0,0) + c.y * A.at<float>(0,1);
        affine_corners[j].y = p.y + c.x * A.at<float>(1,0) + c.y * A.at<float>(1,1);
    }
    draw_quad(img, affine_corners, color);
}

void draw_perspective(Mat &img, Point2f p, int half_width, Mat H, Scalar color){
    const float hw{(float)half_width};
    const vector<Point2f> patch_corners = {p + Point2f{-hw,-hw}, p + Point2f{hw,-hw}, p + Point2f{-hw,hw}, p + Point2f{hw,hw}};
    vector<Point2f> perspective_corners(4);
    for (size_t j = 0; j < 4; j++){
        auto c = patch_corners[j];
        float x = c.x * H.at<float>(0,0) + c.y * H.at<float>(0,1) + H.at<float>(0,2);
        float y = c.x * H.at<float>(1,0) + c.y * H.at<float>(1,1) + H.at<float>(1,2);
        float w = c.x * H.at<float>(2,0) + c.y * H.at<float>(2,1) + H.at<float>(2,2);
        x /= w;
        y /= w;
        perspective_corners[j] = Point2f{x,y};
    }
    draw_quad(img, perspective_corners, color);
}

void draw_transform(Mat &img, Point2f p, int half_width, Mat T, Scalar color){
    if (T.rows == 2)
        draw_affine(img, p, half_width, T, color);
    else if (T.rows == 3)
        draw_perspective(img, p, half_width, T, color);
    else
        throw std::runtime_error("Invalid transform size");
}

void draw_dashed_line(Mat &img, Point2f p0, Point2f p1, Scalar color) {
    cv::LineIterator it(img, p0, p1);
    for(int i = 0; i < it.count; i++,it++){
        if ( i%5!=0 ) // every 5'th pixel gets dropped 
        {
            (*it)[0] = color[0];
            (*it)[1] = color[1];
            (*it)[2] = color[2];
        }         
    }
}
