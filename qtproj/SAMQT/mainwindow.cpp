#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include "QImage"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    std::string encoder_model_path = "/home/arno/workspace/projects/SegmentAnything-OnnxRunner/onnx_models/mobile_sam_encoder.onnx";
    std::string decoder_model_path = "/home/arno/workspace/projects/SegmentAnything-OnnxRunner/onnx_models/mobile_sam_decoder.onnx";
    this->ui->label->InitModel(encoder_model_path,decoder_model_path);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_read_image_clicked()
{
    QImage img("/home/arno/workspace/projects/SegmentAnything-OnnxRunner/test.jpg");


    this->ui->label->SetImage(img);
    this->ui->label->repaint();
}

void MainWindow::on_ckb_boxprompt_stateChanged(int arg1)
{
    this->ui->label->SetBoxPrompt( this->ui->ckb_boxprompt->isChecked());
}
