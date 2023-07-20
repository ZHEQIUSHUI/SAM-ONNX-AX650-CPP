#pragma once
#include "string"
#include "BaseRunner.hpp"
#include "opencv2/opencv.hpp"

#include "SAMEncoder.hpp"

class SAMEncoderOnnx : public SAMEncoder
{
private:
    std::shared_ptr<BaseRunner> model;

public:
    int Load(std::string model_file) override
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

    int Inference(cv::Mat src, float &scale) override
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

    int InputWidth() override
    {
        return model->getInputShape(0)[3];
    }
    int InputHeight() override
    {
        return model->getInputShape(0)[2];
    }

    float *FeaturePtr() override
    {
        return model->getOutputPtr(0);
    }

    int FeatureSize() override
    {
        return nFeatureSize;
    }

    std::vector<unsigned int> FeatureShape() override
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
