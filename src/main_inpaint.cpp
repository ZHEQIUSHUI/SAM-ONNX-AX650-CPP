#include <opencv2/opencv.hpp>
#include "cmdline.hpp"
#include "Runner/LamaInpaintAX650.hpp"
#include "Runner/LamaInpaintOnnx.hpp"
#include "Runner/string_utility.hpp"

int main(int argc, char *argv[])
{
    std::string image_path = "./test.jpg";
    std::string mask_path = "./test.png";
    std::string model_path = "/home/arno/workspace/pycode/lama/big-lama-regular/big-lama-regular.onnx";

    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "model(onnx/axmodel)", true, model_path);
    cmd.add<std::string>("image", 'i', "image file(jpg png etc....)", true, image_path);

    cmd.add<std::string>("mask", 0, "mask file(png etc....)", true, image_path);

    cmd.parse_check(argc, argv);

    image_path = cmd.get<std::string>("image");
    mask_path = cmd.get<std::string>("mask");
    model_path = cmd.get<std::string>("model");
    cv::Mat src = cv::imread(image_path);
    cv::Mat mask = cv::imread(mask_path, cv::ImreadModes::IMREAD_GRAYSCALE);

    printf("image %dx%d\n", src.cols, src.rows);
    printf("mask %dx%d\n", mask.cols, mask.rows);

    std::shared_ptr<LamaInpaint> mInpaint;
    if (string_utility<std::string>::ends_with(model_path, ".onnx"))
    {
        mInpaint.reset(new LamaInpaintOnnx);
    }
    else if (string_utility<std::string>::ends_with(model_path, ".axmodel"))
    {
        mInpaint.reset(new LamaInpaintAX650);
    }
    else
    {
        fprintf(stderr, "no impl for %s\n", model_path.c_str());
        return -1;
    }

    mInpaint->Load(model_path);
    auto time_start = std::chrono::high_resolution_clock::now();
    auto inpainted = mInpaint->Inpaint(src, mask);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "Inpaint Inference Cost time : " << diff.count() << "s" << std::endl;
    cv::imwrite("inpainted.png", inpainted);
    cv::imwrite("mask.png", mask);

    return EXIT_SUCCESS;
}
