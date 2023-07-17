#pragma once

#include "../BaseRunner.hpp"

#include "openvino/openvino.hpp"
#include <fstream>

class OpenvinoRunner : virtual public BaseRunner
{
protected:
    ov::Core core;
    // std::shared_ptr<ov::Model> model;

    std::vector<ov::Output<const ov::Node>> inputs;

    std::vector<ov::Output<const ov::Node>> outputs;

    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

public:
    OpenvinoRunner(/* args */) {}
    ~OpenvinoRunner() {}

    int load(BaseConfig &config) override
    {
        compiled_model = core.compile_model(config.onnx_model);
        inputs = compiled_model.inputs();
        outputs = compiled_model.outputs();
        infer_request = compiled_model.create_infer_request();
        return 0;
    }

    int inference() override
    {
        infer_request.infer();
        return 0;
    }

    int getInputCount() override
    {
        return inputs.size();
    }

    std::vector<size_t> getInputShape(int idx) override
    {
        return inputs[idx].get_shape();
    }

    std::string getInputName(int idx) override
    {
        return inputs[idx].get_any_name();
    }

    float *getInputPtr(int idx) override
    {
        return infer_request.get_input_tensor(idx).data<float>();
    }

    int getOutputCount() override
    {
        return outputs.size();
    }

    std::vector<size_t> getOutputShape(int idx) override
    {
        return outputs[idx].get_shape();
    }

    std::string getOutputName(int idx) override
    {
        return outputs[idx].get_any_name();
    }

    float *getOutputPtr(int idx) override
    {
        return infer_request.get_output_tensor(idx).data<float>();
    }
};
