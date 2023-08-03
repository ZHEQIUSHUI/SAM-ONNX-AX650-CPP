#pragma once
#include "opencv2/opencv.hpp"
#include "LamaInpaint.hpp"
#include "ax_model_runner_ax650.hpp"

class LamaInpaintAX650 : public LamaInpaint
{
private:
    std::shared_ptr<ax_runner_base> model;

public:
    int Load(std::string model_file) override
    {
        model.reset(new ax_runner_ax650);
        model->init(model_file.c_str());
        auto output = model->get_input(0);
        width = output.vShape[3];
        height = output.vShape[2];
        return 0;
    }

    cv::Mat Inpaint(cv::Mat image, cv::Mat mask) override
    {
        cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(11, 11)));
        float scale = get_input_data_letterbox(image, letterbox_img, height, width, false);
        get_input_data_letterbox(mask, letterbox_mask, height, width, false);

        uchar *img_data = letterbox_mask.data;

        for (size_t i = 0; i < height * width; i++)
        {
            img_data[i] = img_data[i] > 127 ? 1.0f : 0.0f;
        }

        memcpy(model->get_input(0).pVirAddr, letterbox_img.data, width * height * 3);
        memcpy(model->get_input(1).pVirAddr, letterbox_mask.data, width * height * 1);

        auto ret = model->inference();

        float *output = (float *)model->get_output(0).pVirAddr;

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
