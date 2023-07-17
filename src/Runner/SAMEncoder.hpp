#pragma once
#include "string"
#include "ax_model_runner_ax650.hpp"
#include "opencv2/opencv.hpp"

class SAMEncoder
{
private:
    std::shared_ptr<ax_runner_base> model;
    // std::vector<float> samfeature;
    int nFeatureSize = 1;
    cv::Mat input;

    float get_input_data_letterbox(cv::Mat mat, cv::Mat&img_new, int letterbox_rows, int letterbox_cols, bool bgr2rgb = true)
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

    int Inference(cv::Mat src, float &scale)
    {
        scale = get_input_data_letterbox(src, input, InputHeight(), InputWidth(), true);
        ax_image_t aximage;
        aximage.nWidth = InputWidth();
        aximage.nHeight = InputHeight();
        aximage.pVir = input.data;
        aximage.tStride_W = aximage.nWidth;

        auto ret = model->inference(&aximage);

        return ret;
    }

    int InputWidth()
    {
        return model->get_algo_width();
    }
    int InputHeight()
    {
        return model->get_algo_height();
    }

    float *FeaturePtr()
    {
        return (float *)model->get_output(0).pVirAddr;
    }

    int FeatureSize()
    {
        return nFeatureSize;
    }

    std::vector<unsigned int> FeatureShape()
    {
        return model->get_output(0).vShape;
    }
};
