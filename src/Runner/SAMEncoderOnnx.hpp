#pragma once
#include "string"
#include "BaseRunner.hpp"
#include "opencv2/opencv.hpp"

class SAMEncoderOnnx
{
private:
    std::shared_ptr<BaseRunner> model;
    // std::vector<float> samfeature;
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
    int Load(std::string model_file)
    {
        model = CreateRunner(RT_OnnxRunner);
        BaseConfig config;
        config.nthread = 4;
        config.onnx_model = model_file;
        model->load(config);
        auto output = model->getOutputShape(0);

        nFeatureSize = 1;
        for (size_t i = 0; i < output.size(); i++)
        {
            nFeatureSize *= output[i];
        }
        return 0;
    }

    int Inference(cv::Mat src, float &scale)
    {
        scale = get_input_data_letterbox(src, input, InputHeight(), InputWidth(), true);

        float *inputPtr = (float *)model->getInputPtr(0);

        uchar *img_data = input.data;

        static float _mean_val[3] = {123.675, 116.28, 103.53};
        static float _std_val[3] = {1.f / 58.395, 1.f / 57.12, 1.f / 57.375};
        int letterbox_cols = InputWidth();
        int letterbox_rows = InputHeight();
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    inputPtr[out_index] = (float(img_data[in_index]) - _mean_val[c]) * _std_val[c];
                }
            }
        }

        auto ret = model->inference();

        return ret;
    }

    int InputWidth()
    {
        return model->getInputShape(0)[3];
    }
    int InputHeight()
    {
        return model->getInputShape(0)[2];
    }

    float *FeaturePtr()
    {
        return model->getOutputPtr(0);
    }

    int FeatureSize()
    {
        return nFeatureSize;
    }

    std::vector<unsigned int> FeatureShape()
    {
        auto shape = model->getOutputShape(0);
        std::vector<unsigned int> out(shape.size());
        for (size_t i = 0; i < shape.size(); i++)
        {
            out[i] = shape[i];
        }
        return out;
    }
};
