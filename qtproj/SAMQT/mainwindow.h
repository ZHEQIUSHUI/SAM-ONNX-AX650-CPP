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
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btn_read_image_clicked();

    void on_ckb_boxprompt_stateChanged(int arg1);

    void on_ckb_realtime_decode_stateChanged(int arg1);

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
