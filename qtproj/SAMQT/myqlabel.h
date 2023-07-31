#ifndef MYQLABEL_H
#define MYQLABEL_H
#include <QLabel>
#include <QPainter>
#include <qimage.h>
#include <QMouseEvent>
#include "libsam/src/Runner/SAM.hpp"

class myQLabel : public QLabel
{
    Q_OBJECT
Q_SIGNALS:
    void mousePressSignal(QMouseEvent *e);
    void mouseMoveSignal(QMouseEvent *e);
    void repaintSignal(QPaintEvent *e);

private:
    QImage cur_image;
    QImage cur_mask;
    cv::Mat rgba_mask;
    std::vector<MatInfo> v_mask;
    bool isBoxPrompt = false;
    bool isRealtimeDecode = false;
    bool mouseHolding = false;
    QPoint pt_img_first, pt_img_secend;
    SAM sam;

    void mousePressEvent(QMouseEvent *e) override
    {
        pt_img_first = pt_img_secend = getSourcePoint(this->size(), cur_image.size(), e->pos());
        mouseHolding = true;
        repaint();
    }

    void mouseReleaseEvent(QMouseEvent *e) override
    {
        pt_img_secend = getSourcePoint(this->size(), cur_image.size(), e->pos());
        mouseHolding = false;
        samDecode();
        repaint();
    }

    void mouseMoveEvent(QMouseEvent *e) override
    {
        if (mouseHolding)
        {
            pt_img_secend = getSourcePoint(this->size(), cur_image.size(), e->pos());
            if (isRealtimeDecode)
                samDecode();
            repaint();
        }
    }

    void samDecode()
    {
        if (isBoxPrompt)
        {
            v_mask = sam.Decode(cv::Rect(cv::Point(pt_img_first.x(), pt_img_first.y()), cv::Point(pt_img_secend.x(), pt_img_secend.y())));
        }
        else
        {
            v_mask = sam.Decode(cv::Point(pt_img_secend.x(), pt_img_secend.y()));
        }
        int maxid = -1;
        float max_score = 0;
        for (size_t i = 0; i < v_mask.size(); i++)
        {
            if (v_mask[i].iou_pred > max_score)
            {
                max_score = v_mask[i].iou_pred;
                maxid = i;
            }
        }

        rgba_mask = cv::Mat(v_mask[maxid].mask.rows, v_mask[maxid].mask.cols, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        rgba_mask.setTo(cv::Scalar(200, 200, 0, 200), v_mask[maxid].mask);
        cur_mask = QImage(rgba_mask.data, rgba_mask.cols, rgba_mask.rows, QImage::Format_RGBA8888);
    }

    QPoint getSourcePoint(QSize window, QSize img, QPoint pt)
    {
        float letterbox_rows = window.height();
        float letterbox_cols = window.width();
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / img.height()) < (letterbox_cols * 1.0 / img.width()))
        {
            scale_letterbox = (float)letterbox_rows * 1.0f / (float)img.height();
        }
        else
        {
            scale_letterbox = (float)letterbox_cols * 1.0f / (float)img.width();
        }
        resize_cols = int(scale_letterbox * (float)img.width());
        resize_rows = int(scale_letterbox * (float)img.height());
        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;
        float ratio_x = (float)img.height() / resize_rows;
        float ratio_y = (float)img.width() / resize_cols;
        auto x0 = pt.x();
        auto y0 = pt.y();
        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        return QPoint(x0, y0);
    }

    QPoint getWindowPoint(QSize window, QSize img, QPoint pt)
    {
        float letterbox_rows = window.height();
        float letterbox_cols = window.width();
        float scale_letterbox;
        int resize_rows;
        int resize_cols;
        if ((letterbox_rows * 1.0 / img.height()) < (letterbox_cols * 1.0 / img.width()))
        {
            scale_letterbox = (float)letterbox_rows * 1.0f / (float)img.height();
        }
        else
        {
            scale_letterbox = (float)letterbox_cols * 1.0f / (float)img.width();
        }
        resize_cols = int(scale_letterbox * (float)img.width());
        resize_rows = int(scale_letterbox * (float)img.height());
        int tmp_h = (letterbox_rows - resize_rows) / 2;
        int tmp_w = (letterbox_cols - resize_cols) / 2;
        float ratio_x = (float)img.height() / resize_rows;
        float ratio_y = (float)img.width() / resize_cols;
        auto x0 = pt.x();
        auto y0 = pt.y();
        x0 = x0 / ratio_x + tmp_w;
        y0 = y0 / ratio_y + tmp_h;
        return QPoint(x0, y0);
    }

    QRect getTargetRect(QImage img)
    {
        return QRect(QPoint(getWindowPoint(this->size(), img.size(), {0, 0})), QPoint(getWindowPoint(this->size(), img.size(), {img.width(), img.height()})));
    }

    void paintEvent(QPaintEvent *event) override
    {
        QPainter p(this);
        p.drawImage(getTargetRect(cur_image), cur_image);
        p.drawImage(getTargetRect(cur_image), cur_mask);
        QColor color(0, 255, 0, 200);
        p.setPen(QPen(color, 3));
        if (isBoxPrompt)
        {
            p.setPen(QPen(color, 3));
            p.drawRect(QRect(getWindowPoint(this->size(), cur_image.size(), pt_img_first), getWindowPoint(this->size(), cur_image.size(), pt_img_secend)));
        }
        else
        {
            p.drawEllipse(getWindowPoint(this->size(), cur_image.size(), pt_img_secend), 2, 2);
        }
    }

public:
    myQLabel(QWidget *parent) : QLabel(parent)
    {
    }

    void SetImage(QImage img)
    {
        cur_mask = QImage();
        pt_img_first = QPoint(-10000, -10000);
        pt_img_secend = QPoint(-10000, -10000);
        cur_image = img;
        cv::Mat src(cur_image.height(), cur_image.width(), CV_8UC4, cur_image.bits());
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_RGBA2RGB);
        sam.Encode(rgb);
        repaint();
    }

    void SetBoxPrompt(bool useBoxprompt)
    {
        isBoxPrompt = useBoxprompt;
        cur_mask = QImage();
        pt_img_first = QPoint(-10000, -10000);
        pt_img_secend = QPoint(-10000, -10000);
        repaint();
    }

    void SetRealtimeDecode(bool RealtimeDecode)
    {
        isRealtimeDecode = RealtimeDecode;
    }

    void InitModel(std::string encoder_model, std::string decoder_model)
    {
        sam.Load(encoder_model, decoder_model);
    }
};
#endif // MYQLABEL_H
