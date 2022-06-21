
#include <opencv2/core/core.hpp>
#include <net.h>

#ifndef NCNN_ANDROID_MEDIAPIPE_BLAZEFACE_CARDDETECT_H
#define NCNN_ANDROID_MEDIAPIPE_BLAZEFACE_CARDDETECT_H

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

struct CenterPrior
{
    int x;
    int y;
    int stride;
};

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};

class CardDetect
{
public:
    int load(AAssetManager* mgr, bool use_gpu = false);
    int detect(const cv::Mat& rgb, std::vector<BoxInfo> &objects, float score_threshold, float nms_threshold);

private:
    void preprocess(const cv::Mat& image, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& feats, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);

    ncnn::Net card;
    int num_class = 2;
    const int input_size[2] = {416, 416};
    int reg_max = 2;
    std::vector<int> strides = { 8, 16, 32, 64 };
};

#endif //NCNN_ANDROID_MEDIAPIPE_BLAZEFACE_CARDDETECT_H

