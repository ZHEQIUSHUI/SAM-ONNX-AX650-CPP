#pragma once
#include "string"
#include "ax_model_runner_ax650.hpp"
#include "opencv2/opencv.hpp"

class SAMEncoder
{
protected:
    int nFeatureSize = 1;
    cv::Mat input;

    float get_input_data_letterbox(cv::Mat mat, cv::Mat &img_new, int letterbox_rows, int letterbox_cols, bool bgr2rgb = true)
    {
        /* letterbox process to support different letterbox size */
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / mat.rows) < (letterbox_cols * 1.0 / mat.cols))
        {
            scale_letterbox = (float)letterbox_rows * 1.0f / (float)mat.rows;
        }
        else
        {
            scale_letterbox = (float)letterbox_cols * 1.0f / (float)mat.cols;
        }
        resize_cols = int(scale_letterbox * (float)mat.cols);
        resize_rows = int(scale_letterbox * (float)mat.rows);

        cv::resize(mat, mat, cv::Size(resize_cols, resize_rows));

        int top = (letterbox_rows - resize_rows) / 2;
        int bot = (letterbox_rows - resize_rows + 1) / 2;
        int left = (letterbox_cols - resize_cols) / 2;
        int right = (letterbox_cols - resize_cols + 1) / 2;

        // Letterbox filling
        cv::copyMakeBorder(mat, img_new, 0, top + bot, 0, left + right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        if (bgr2rgb)
        {
            cv::cvtColor(img_new, img_new, cv::COLOR_BGR2RGB);
        }
        return scale_letterbox;
    }

public:
    virtual int Load(std::string model_file) = 0;

    virtual int Inference(cv::Mat src, float &scale) = 0;
    virtual int InputWidth() = 0;
    virtual int InputHeight() = 0;

    virtual float *FeaturePtr() = 0;
    virtual int FeatureSize() = 0;
    virtual std::vector<unsigned int> FeatureShape() = 0;
};
