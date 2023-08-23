#include "mainwindow.h"

#include <QApplication>

#include "style/DarkStyle.h"
#include "libsam/src/cmdline.hpp"

int main(int argc, char *argv[])
{
    std::string encoder_model_path = "/root/qtproj/SAM-ONNX-AX650-CPP/ax_models/sam-encoder.axmodel";
    std::string decoder_model_path = "/root/qtproj/SAM-ONNX-AX650-CPP/ax_models/sam_vit_b_01ec64_decoder.onnx";
    std::string inpaint_model_path = "/root/qtproj/SAM-ONNX-AX650-CPP/ax_models/big-lama-regular.axmodel";


    cmdline::parser cmd;
    cmd.add<std::string>("encoder", 'e', "encoder model(onnx model or axmodel)", true, encoder_model_path);
    cmd.add<std::string>("decoder", 'd', "decoder model(onnx)", true, decoder_model_path);
    cmd.add<std::string>("inpaint", 'i', "inpaint model(onnx)", true, inpaint_model_path);

    cmd.parse_check(argc, argv);

    encoder_model_path = cmd.get<std::string>("encoder");
    decoder_model_path = cmd.get<std::string>("decoder");
    inpaint_model_path = cmd.get<std::string>("inpaint");

    QApplication a(argc, argv);
    QApplication::setStyle(new DarkStyle);
    MainWindow w(encoder_model_path,decoder_model_path,inpaint_model_path);
    w.show();
    return a.exec();
}
