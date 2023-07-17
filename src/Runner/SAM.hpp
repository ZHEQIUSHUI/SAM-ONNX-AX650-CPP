#pragma once

#include "SAMDecoderOnnx.hpp"
#include "SAMEncoderOnnx.hpp"
#include "SAMEncoder.hpp"

#include "string_utility.hpp"
class SAM
{
private:
    SAMDecoderOnnx decoder;
    SAMEncoder encoder;
    SAMEncoderOnnx encoder_onnx;

    bool b_onnx = false;

public:
    SAM() {}
    int Load(std::string encoder_model, std::string decoder_model)
    {
        if (string_utility<std::string>::ends_with(encoder_model, ".onnx"))
        {
            encoder_onnx.Load(encoder_model);
            b_onnx = true;
        }
        else if (string_utility<std::string>::ends_with(encoder_model, ".axmodel"))
        {
            encoder.Load(encoder_model);
            b_onnx = false;
        }
        else
        {
            ALOGE("");
            return -1;
        }

        decoder.Load(decoder_model);
        return 0;
    }

    void Encode(cv::Mat src)
    {
        float scale;
        if (b_onnx)
        {
            auto time_start = std::chrono::high_resolution_clock::now();
            encoder_onnx.Inference(src, scale);
            auto time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = time_end - time_start;
            std::cout << "Encoder Inference Cost time : " << diff.count() << "s" << std::endl;

            decoder.LoadFeature(src.cols, src.rows, encoder_onnx.FeaturePtr(), encoder_onnx.FeatureSize(), encoder_onnx.FeatureShape(), scale);
        }
        else
        {
            auto time_start = std::chrono::high_resolution_clock::now();
            encoder.Inference(src, scale);
            auto time_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = time_end - time_start;
            std::cout << "Encoder Inference Cost time : " << diff.count() << "s" << std::endl;

            decoder.LoadFeature(src.cols, src.rows, encoder.FeaturePtr(), encoder.FeatureSize(), encoder.FeatureShape(), scale);
        }
    }

    std::vector<MatInfo> Decode(cv::Point pt)
    {
        return decoder.Inference(pt);
    }

    std::vector<MatInfo> Decode(cv::Rect box)
    {
        return decoder.Inference(box);
    }
};
