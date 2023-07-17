#include "BaseRunner.hpp"
#include "OnnxWarpper/OnnxWarpper.hpp"

std::shared_ptr<BaseRunner> CreateRunner(RunnerType rt)
{
    switch (rt)
    {
    case RT_OnnxRunner:
        return std::make_shared<OnnxRunner>();
    case RT_OpenvinoRunner:
    case RT_TensorrtRunner:
    default:
        return nullptr;
    }
    return nullptr;
}
