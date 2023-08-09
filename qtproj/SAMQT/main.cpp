#include "mainwindow.h"

#include <QApplication>

#include "style/DarkStyle.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QApplication::setStyle(new DarkStyle);
    MainWindow w;
    w.show();
    return a.exec();
}
