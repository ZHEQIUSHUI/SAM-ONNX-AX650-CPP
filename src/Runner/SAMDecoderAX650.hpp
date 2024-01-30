#pragma once
#include "SAMDecoderOnnx.hpp"
#include "ax_model_runner_ax650.hpp"

class SAMDecoderAX650V2 : public SAMDecoderOnnxV2
{
protected:
    std::shared_ptr<ax_runner_base> DecoderSession;

    std::vector<MatInfo> Decoder_Inference(cv::Point *clickinfo, cv::Rect *boxinfo) override
    {
        auto DecoderEncPtsOutputTensors = EncPts(clickinfo, boxinfo);

        memcpy(DecoderSession->get_input(0).pVirAddr, samfeature.data(), 1 * 256 * 64 * 64 * sizeof(float));
        memcpy(DecoderSession->get_input(1).pVirAddr, DecoderEncPtsOutputTensors[0].GetTensorMutableData<float>(), 1 * 3 * 256 * sizeof(float));

        DecoderSession->inference();
        float *iou_predictions = (float *)DecoderSession->get_output(0).pVirAddr;
        float *masks = (float *)DecoderSession->get_output(1).pVirAddr;

        return PostPorcess(iou_predictions, masks);
    }

public:
public:
    int Load(std::string model_file_pts, std::string model_file_sub, int nthread = 4) override
    {
        // 初始化OnnxRuntime运行环境
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SegmentAnythingModel");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        // 设置图像优化级别
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // DecoderSession.reset(new Ort::Session(env, model_file_sub.c_str(), session_options));
        // if (DecoderSession->GetInputCount() != 2 || DecoderSession->GetOutputCount() != 2)
        // {
        //     ALOGE("DecoderSession Model not loaded (invalid input/output count)");
        //     return -1;
        // }
        DecoderSession.reset(new ax_runner_ax650);
        int ret = DecoderSession->init(model_file_sub.c_str());
        if (ret != 0)
        {
            return -1;
        }

        if (DecoderSession->get_num_inputs() != 2 || DecoderSession->get_num_outputs() != 2)
        {
            ALOGE("DecoderSession Model not loaded (invalid input/output count)");
            return -1;
        }

        DecoderEncPtsSession.reset(new Ort::Session(env, model_file_pts.c_str(), session_options));
        if (DecoderEncPtsSession->GetInputCount() != 2 || DecoderEncPtsSession->GetOutputCount() != 1)
        {
            ALOGE("DecoderEncPtsSession Model not loaded (invalid input/output count)");
            return -1;
        }
        return 0;
    }
};
