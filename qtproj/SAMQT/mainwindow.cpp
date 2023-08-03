#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include "QImage"
#include "QFileDialog"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    std::string encoder_model_path = "/home/arno/workspace/projects/SegmentAnything-OnnxRunner/onnx_models/mobile_sam_encoder.onnx";
    std::string decoder_model_path = "/home/arno/workspace/projects/SegmentAnything-OnnxRunner/onnx_models/mobile_sam_decoder.onnx";
    std::string inpaint_model_path = "/home/arno/workspace/pycode/lama/big-lama-regular/big-lama-regular.onnx";
    this->ui->label->InitModel(encoder_model_path, decoder_model_path,inpaint_model_path);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_btn_read_image_clicked()
{
    auto filename = QFileDialog::getOpenFileName(this, "", "", "image(*.png *.jpg *.jpeg *.bmp)");
    if (filename.isEmpty())
    {
        return;
    }
    QImage img(filename);
    this->ui->label->SetImage(img);
}

void MainWindow::on_ckb_boxprompt_stateChanged(int arg1)
{
    this->ui->label->SetBoxPrompt(this->ui->ckb_boxprompt->isChecked());
}

void MainWindow::on_ckb_realtime_decode_stateChanged(int arg1)
{
    this->ui->label->SetRealtimeDecode(this->ui->ckb_realtime_decode->isChecked());
}

void MainWindow::on_btn_remove_obj_clicked()
{
    this->ui->label->ShowRemoveObject();
}

void MainWindow::on_btn_reset_clicked()
{
    this->ui->label->Reset();
}
