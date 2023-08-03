#pragma once
#include "BaseRunner.hpp"
#include "opencv2/opencv.hpp"
#include "LamaInpaint.hpp"

class LamaInpaintOnnx : public LamaInpaint
{
protected:
    std::shared_ptr<BaseRunner> model;

public:
    int Load(std::string model_file) override
    {
        model = CreateRunner(RT_OnnxRunner);
        BaseConfig config;
        config.nthread = 4;
        config.onnx_model = model_file;
        model->load(config);
        auto output = model->getInputShape(0);
        width = output[3];
        height = output[2];
        return 0;
    }

    cv::Mat Inpaint(cv::Mat image, cv::Mat mask) override
    {
        cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(11, 11)));
        float scale = get_input_data_letterbox(image, letterbox_img, height, width, false);
        get_input_data_letterbox(mask, letterbox_mask, height, width, false);

        float *inputPtr = (float *)model->getInputPtr(0);

        uchar *img_data = letterbox_img.data;

        int letterbox_cols = width;
        int letterbox_rows = height;
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    inputPtr[out_index] = img_data[in_index] * (1 / 255.f);
                }
            }
        }

        inputPtr = (float *)model->getInputPtr(1);
        img_data = letterbox_mask.data;

        for (size_t i = 0; i < height * width; i++)
        {
            inputPtr[i] = img_data[i] > 127 ? 1.0f : 0.0f;
        }

        auto ret = model->inference();

        float *output = model->getOutputPtr(0);

        cv::Mat output_image(height, width, CV_8UC3);

        uchar *output_img_data = output_image.data;

        for (size_t i = 0; i < height * width * 3; i++)
        {
            output_img_data[i] = MAX(0, MIN(255, output[i]));
        }

        int output_height = MIN(output_image.rows, image.rows * scale);
        int output_width = MIN(output_image.cols, image.cols * scale);

        auto crop = output_image(cv::Rect(0, 0, output_width, output_height));

        cv::resize(crop, crop, cv::Size(image.cols, image.rows));
        return crop;
    }
};
