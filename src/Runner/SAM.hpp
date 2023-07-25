#pragma once

#include "SAMDecoderOnnx.hpp"
#include "SAMEncoderOnnx.hpp"
#include "SAMEncoderAX650.hpp"

#include "string_utility.hpp"
class SAM
{
private:
    SAMDecoderOnnx decoder;
    std::shared_ptr<SAMEncoder> encoder;
    bool bInit = false;

public:
    SAM() {}
    int Load(std::string encoder_model, std::string decoder_model)
    {
        if (string_utility<std::string>::ends_with(encoder_model, ".onnx"))
        {
            encoder.reset(new SAMEncoderOnnx);
        }
        else if (string_utility<std::string>::ends_with(encoder_model, ".axmodel"))
        {
            encoder.reset(new SAMEncoderAX650);
        }
        else
        {
            ALOGE("no impl for %s", encoder_model.c_str());
            return -1;
        }
        encoder->Load(encoder_model);
        decoder.Load(decoder_model);
        bInit = true;
        return 0;
    }

    void Encode(cv::Mat src)
    {
        if (!bInit)
        {
            return;
        }

        float scale;
        auto time_start = std::chrono::high_resolution_clock::now();
        encoder->Inference(src, scale);
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = time_end - time_start;
        std::cout << "Encoder Inference Cost time : " << diff.count() << "s" << std::endl;

        decoder.LoadFeature(src.cols, src.rows, encoder->FeaturePtr(), encoder->FeatureSize(), encoder->FeatureShape(), scale);
    }

    std::vector<MatInfo> Decode(cv::Point pt)
    {
        if (!bInit)
        {
            return std::vector<MatInfo>();
        }
        return decoder.Inference(pt);
    }

    std::vector<MatInfo> Decode(cv::Rect box)
    {
        if (!bInit)
        {
            return std::vector<MatInfo>();
        }
        return decoder.Inference(box);
    }
};
