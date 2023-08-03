#pragma once
#include "string"
#include "ax_model_runner_ax650.hpp"
#include "opencv2/opencv.hpp"

#include "SAMEncoder.hpp"

class SAMEncoderAX650 : public SAMEncoder
{
private:
    std::shared_ptr<ax_runner_base> model;

public:
    int Load(std::string model_file) override
    {
        model.reset(new ax_runner_ax650);
        model->init(model_file.c_str());
        auto &output = model->get_output(0);

        nFeatureSize = 1;
        for (size_t i = 0; i < output.vShape.size(); i++)
        {
            nFeatureSize *= output.vShape[i];
        }
        return 0;
    }

    int Inference(cv::Mat src, float &scale) override
    {
        scale = get_input_data_letterbox(src, input, InputHeight(), InputWidth(), true);
        memcpy(model->get_input(0).pVirAddr, input.data, input.cols * input.rows * 3);
        auto ret = model->inference();
        return ret;
    }

    int InputWidth() override
    {
        return model->get_algo_width();
    }
    int InputHeight() override
    {
        return model->get_algo_height();
    }

    float *FeaturePtr() override
    {
        return (float *)model->get_output(0).pVirAddr;
    }

    int FeatureSize() override
    {
        return nFeatureSize;
    }

    std::vector<unsigned int> FeatureShape() override
    {
        return model->get_output(0).vShape;
    }
};
