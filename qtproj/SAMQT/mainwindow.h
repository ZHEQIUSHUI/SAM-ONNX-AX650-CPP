#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "myqlabel.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(std::string encoder_model_path, std::string decoder_model_path, std::string inpaint_model_path, QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btn_read_image_clicked();

    void on_ckb_realtime_decode_stateChanged(int arg1);

    void on_btn_remove_obj_clicked();

    void on_btn_reset_clicked();

    void on_radioButton_point_clicked();

    void on_radioButton_box_clicked();

    void on_btn_save_img_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
