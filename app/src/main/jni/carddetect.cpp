#include "carddetect.h"

#include <string.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}


void CardDetect::decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results)
{
    const int num_points = center_priors.size();
    //printf("num_points:%d\n", num_points);

    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors[idx].x;
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        const float* scores = feats.row(idx);
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < this->num_class; label++)
        {
            if (scores[label] > score)
            {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold)
        {
            //std::cout << "label:" << cur_label << " score:" << score << std::endl;
            const float* bbox_pred = feats.row(idx) + this->num_class;
            results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            //cv::imshow("debug", debug_heatmap);
        }
    }
}

BoxInfo CardDetect::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[this->reg_max + 1];
        activation_function_softmax(dfl_det + i * (this->reg_max + 1), dis_after_sm, this->reg_max + 1);
        for (int j = 0; j < this->reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size[0]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size[1]);

    //std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}

void CardDetect::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else {
                j++;
            }
        }
    }
}


void CardDetect::preprocess(const cv::Mat& image, ncnn::Mat& in)
{
    int img_w = image.cols;
    int img_h = image.rows;

    in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, img_w, img_h);
    in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w, img_h, input_size[1], input_size[0]);

    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
    in.substract_mean_normalize(mean_vals, norm_vals);
}

int CardDetect::load(AAssetManager* mgr, bool use_gpu)
{
    card.clear();
    card.opt = ncnn::Option();
#if NCNN_VULKAN
    card.opt.use_vulkan_compute = use_gpu;
#endif

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    card.opt.num_threads = ncnn::get_big_cpu_count();
    card.load_param(mgr, "nonodet-cdcard-opt.param");
    card.load_model(mgr, "nanodet-cdcard-opt.bin");
//    card.load_param(mgr, "credit-card-2500s.param");
//    card.load_model(mgr, "credit-card-2500s.bin");

    return 0;
}

void refine_card(const cv::Mat& rgb, const BoxInfo boxInfo, const cv::Mat& main_rgb, std::vector<cv::Point2f>& rect_points2)
{
    BoxInfo box = boxInfo;
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "box1=%.f,%.f,%.f,%.f", box.x1, box.y1, box.x2, box.y2);

    int width = cv::abs(box.x2 - box.x1);
    int height = cv::abs(box.y2 - box.y1);
    float inflate_w = width * 0.1;
    float inflate_h = height * 0.1;
    //inflate_w = 0;
    //inflate_h = 0;
    box.x1 -= inflate_w; if (box.x1 < 0) box.x1 = 0;
    box.y1 -= inflate_h; if (box.y1 < 0) box.y1 = 0;
    box.x2 += inflate_w;
    box.y2 += inflate_h;
    width = cv::abs(box.x2 - box.x1) - 1;
    height = cv::abs(box.y2 - box.y1) - 1;
    if (width <= 0 || height <= 0)
        return;
    if (box.x1 + width >= main_rgb.cols)
        width = main_rgb.cols - box.x1 - 1;
    if (box.y1 + height >= main_rgb.rows)
        height = main_rgb.rows - box.y1 - 1;
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "box2=%.f,%.f,%.f,%.f", box.x1, box.y1, box.x2, box.y2);
    cv::Mat image_card = main_rgb(cv::Rect((int)box.x1, (int)box.y1, width, height));
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "image_card=%d,%d", image_card.cols, image_card.rows);
    cv::Mat image_resized, image_gray, image;
    //float scale = 200 / image_card.cols;
    //resize(image_card, image_resized, cv::Size(), scale, scale, cv::INTER_CUBIC);
    cv::cvtColor(image_card, image_gray, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur(image_gray, image, cv::Size(3, 3), 0);
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "image_gray=%d,%d", image_gray.cols, image_gray.rows);


    cv::Mat clone = image.clone();
    clone = clone.reshape(1, (clone.cols * clone.rows));
    clone.convertTo(clone, CV_32F);
    std::vector<int> labels;
    cv::Mat1f centers;
    cv::kmeans(clone, 3 ,labels,
               cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 2, 1.),
               10,cv::KMEANS_PP_CENTERS,centers);

    double minVal;
    double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

    cv::minMaxLoc( centers, &minVal, &maxVal, &minLoc, &maxLoc );
    int magnetic_index = minLoc.y;
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "magnetic_index=%d", magnetic_index);

    //cv::Mat1b mask = cv::Mat(labels).reshape(image.cols, image.rows);
    cv::Mat1b mask = cv::Mat(labels);
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "mask=%d,%d", mask.cols, mask.rows);

    mask = mask.setTo(255, mask != magnetic_index);
    mask.create(image.rows, image.cols);
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "mask=%d,%d", mask.cols, mask.rows);

    cv::Mat canny;
    cv::Canny(mask, canny, 50, 255, 3);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat temp_color;
    cv::cvtColor(canny, temp_color, cv::COLOR_GRAY2RGB);
    temp_color.copyTo(main_rgb(cv::Rect(0,0, width, height)));

    std::vector<std::vector<cv::Point>> cnts_large;
    for (const auto& c : contours) {
        cv::Rect bbox = cv::boundingRect(c);
        if (bbox.width > mask.cols / 3 || bbox.height > mask.rows / 2)
            cnts_large.push_back(c);
    }

    std::vector<cv::Point> cnts_combined, hull_combined;
    for (auto c : cnts_large) {
        cnts_combined.insert(cnts_combined.end(), c.begin(), c.end());
    }
    //__android_log_print(ANDROID_LOG_INFO, "carddetect", "cnts_combined=%d", cnts_combined.size());

    if (!cnts_combined.empty()) {
        cv::convexHull(cnts_combined, hull_combined, false);
        cv::RotatedRect rect = cv::minAreaRect(hull_combined);
        cv::Point2f rect_points[4];
        rect.points(rect_points);
        int i = 0;
        for (auto p: rect_points) {
            p.x = p.x + box.x1;
            p.y = p.y + box.y1;
            //__android_log_print(ANDROID_LOG_INFO, "carddetect", "rect_points=%f,%f", p.x, p.y);
            rect_points2.push_back(p);
        }
        for (auto p: rect_points2) {
            //__android_log_print(ANDROID_LOG_INFO, "carddetect", "rect_points=%f,%f", p.x, p.y);
        }
//        std::vector<std::vector<cv::Point2f>> data(1);
//        data[0].push_back(rect_points2[0]);
//        data[0].push_back(rect_points2[1]);
//        data[0].push_back(rect_points2[2]);
//        data[0].push_back(rect_points2[3]);
        //cv::drawContours(main_rgb, data, 0, cv::Scalar(0, 255, 255), 2);
        //image_card.copyTo(main_rgb(cv::Rect(0,0, width, height)));

    }
}

cv::Point2f CardDetect::inv_transform(cv::Point_<float> src, float ratio_w, float ratio_h, cv::Mat trans_mat)
{
    cv::Point2f dst;
    float x = src.x * ratio_w;
    float y = src.y * ratio_h;
    dst.x = x * trans_mat.at<double>(0, 0) + y * trans_mat.at<double>(0, 1) + trans_mat.at<double>(0, 2);
    dst.y = x * trans_mat.at<double>(1, 0) + y * trans_mat.at<double>(1, 1) + trans_mat.at<double>(1, 2);
    return dst;
}
int CardDetect::detect(const cv::Mat& rgb, const cv::Mat& trans_mat, std::vector<BoxInfo> &objects, std::vector<cv::Point2f>& rect_points,
                       float score_threshold, float nms_threshold, const cv::Mat& main_rgb)
{
    int src_w = rgb.cols;
    int src_h = rgb.rows;
    int dst_w = input_size[1];
    int dst_h = input_size[0];
    float width_ratio = (float)src_w / (float)dst_w;
    float height_ratio = (float)src_h / (float)dst_h;

    ncnn::Mat input, resized_input;
    preprocess(rgb, input);
    auto ex = card.create_extractor();
    ex.set_light_mode(false);
    ex.set_num_threads(4);
#if NCNN_VULKAN
    //ex.set_vulkan_compute(false);
#endif
    ex.input("data", input);

    std::vector<std::vector<BoxInfo>> results;
    results.resize(this->num_class);

    ncnn::Mat out;
    ex.extract("output", out);
    // printf("%d %d %d \n", out.w, out.h, out.c);

    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> center_priors;
    generate_grid_center_priors(input_size[0], input_size[1], this->strides, center_priors);

    this->decode_infer(out, center_priors, score_threshold, results);

    for (auto & result : results)
    {
        nms(result, nms_threshold);
        for (auto box : result)
        {
            cv::Point2f pt = inv_transform(cv::Point2f(box.x1, box.y1), width_ratio, height_ratio, trans_mat);
            box.x1 = pt.x;
            box.y1 = pt.y;

            pt = inv_transform(cv::Point2f(box.x2, box.y2), width_ratio, height_ratio, trans_mat);
            box.x2 = pt.x;
            box.y2 = pt.y;
            //__android_log_print(ANDROID_LOG_INFO, "carddetect", "box=%.f,%.f,%.f,%.f", box.x1, box.y1, box.x2, box.y2);
            refine_card(rgb, box, main_rgb, rect_points);

            objects.push_back(box);
        }
    }

    return 0;
}
