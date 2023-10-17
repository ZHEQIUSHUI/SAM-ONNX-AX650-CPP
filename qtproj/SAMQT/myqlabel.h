#ifndef MYQLABEL_H
#define MYQLABEL_H
#include <QLabel>
#include <QPainter>
#include <qimage.h>
#include <QMouseEvent>
#include "QDialog"
#include "QMimeData"
#include "QProgressBar"
#include "libsam/src/Runner/SAM.hpp"
#include "libsam/src/Runner/LamaInpaintOnnx.hpp"
#include "libsam/src/Runner/LamaInpaintAX650.hpp"

class myQLabel : public QLabel
{
    Q_OBJECT
Q_SIGNALS:
    void mousePressSignal(QMouseEvent *e);
    void mouseMoveSignal(QMouseEvent *e);
    void repaintSignal(QPaintEvent *e);

private:
    QImage bak_image;
    QImage cur_image;

    QVector<QImage> cur_masks;
    cv::Scalar cur_color_preview;
    QImage cur_mask_preview;
    QVector<cv::Mat> rgba_masks;
    QVector<cv::Mat> grab_masks;

    std::vector<MatInfo> v_mask;
    bool isBoxPrompt = false;
    bool isRealtimeDecode = false;
    bool mouseHolding = false;
    QPoint pt_img_first, pt_img_secend;
    SAM mSam;
    //    LamaInpaintOnnx mInpaint;
    std::shared_ptr<LamaInpaint> mInpaint;

    void dragEnterEvent(QDragEnterEvent *event) override
    {
        if (event->mimeData()->hasUrls())
            event->acceptProposedAction();
        else
            event->ignore();
    }
    void dropEvent(QDropEvent *event) override
    {
        const QMimeData *mimeData = event->mimeData();

        if (!mimeData->hasUrls())
        {
            return;
        }

        QList<QUrl> urlList = mimeData->urls();

        QString filename = urlList.at(0).toLocalFile();
        if (filename.isEmpty())
        {
            return;
        }
        printf("open from drop:%s\n", filename.toStdString().c_str());
        QImage img(filename);
        if (img.bits())
            SetImage(img);
    }

    void mousePressEvent(QMouseEvent *e) override
    {
        if (e->button() == Qt::RightButton)
        {
        }
        else if (e->button() == Qt::LeftButton)
        {
            cur_color_preview = cv::Scalar(rand() % 255, rand() % 255, rand() % 255, 200);
            pt_img_first = pt_img_secend = getSourcePoint(this->size(), cur_image.size(), e->pos());
            mouseHolding = true;
            repaint();
        }
    }

    void mouseReleaseEvent(QMouseEvent *e) override
    {
        if (e->button() == Qt::RightButton)
        {
            pt_img_first = QPoint(-10000, -10000);
            pt_img_secend = QPoint(-10000, -10000);
            RemoveLastMask();
            repaint();
        }
        else if (e->button() == Qt::LeftButton)
        {
            pt_img_secend = getSourcePoint(this->size(), cur_image.size(), e->pos());
            mouseHolding = false;
            samDecode(true);
            repaint();
        }
    }

    void mouseMoveEvent(QMouseEvent *e) override
    {
        if (mouseHolding)
        {
            pt_img_secend = getSourcePoint(this->size(), cur_image.size(), e->pos());
            if (isRealtimeDecode)
                samDecode(false);
            repaint();
        }
    }

    void samDecode(bool push = true)
    {
        if (cur_image.isNull())
            return;
        if (isBoxPrompt)
        {
            v_mask = mSam.Decode(cv::Rect(cv::Point(pt_img_first.x(), pt_img_first.y()), cv::Point(pt_img_secend.x(), pt_img_secend.y())));
        }
        else
        {
            v_mask = mSam.Decode(cv::Point(pt_img_secend.x(), pt_img_secend.y()));
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
        auto grab_mask = v_mask[maxid].mask.clone();
        auto rgba_mask = cv::Mat(v_mask[maxid].mask.rows, v_mask[maxid].mask.cols, CV_8UC4, cv::Scalar(0, 0, 0, 0));
        rgba_mask.setTo(cur_color_preview, v_mask[maxid].mask);
        auto cur_mask = QImage(rgba_mask.data, rgba_mask.cols, rgba_mask.rows, QImage::Format_RGBA8888);

        if (push)
        {
            grab_masks.push_back(grab_mask);
            rgba_masks.push_back(rgba_mask);
            cur_masks.push_back(cur_mask);
            cur_mask_preview = QImage();
        }
        else
        {
            cur_mask_preview = cur_mask.copy();
        }
    }

    QRect getTargetRect(QImage img)
    {
        return QRect(QPoint(getWindowPoint(this->size(), img.size(), {0, 0})), QPoint(getWindowPoint(this->size(), img.size(), {img.width(), img.height()})));
    }

    void paintEvent(QPaintEvent *event) override
    {
        QPainter p(this);
        p.drawImage(getTargetRect(cur_image), cur_image);
        for (auto cur_mask : cur_masks)
            p.drawImage(getTargetRect(cur_image), cur_mask);
        if (cur_mask_preview.bits())
            p.drawImage(getTargetRect(cur_image), cur_mask_preview);
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

    QImage getCurrentImage()
    {
        return cur_image;
    }

    void SetImage(QImage img)
    {
        cur_masks.clear();
        rgba_masks.clear();
        grab_masks.clear();
        pt_img_first = QPoint(-10000, -10000);
        pt_img_secend = QPoint(-10000, -10000);
        cur_image = img;
        bak_image = img.copy();
        cv::Mat src(cur_image.height(), cur_image.width(), CV_8UC4, cur_image.bits());
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_RGBA2RGB);
        mSam.Encode(rgb);
        repaint();
    }

    void SetBoxPrompt(bool useBoxprompt)
    {
        isBoxPrompt = useBoxprompt;
        cur_masks.clear();
        rgba_masks.clear();
        grab_masks.clear();
        pt_img_first = QPoint(-10000, -10000);
        pt_img_secend = QPoint(-10000, -10000);
        repaint();
    }

    void SetRealtimeDecode(bool RealtimeDecode)
    {
        isRealtimeDecode = RealtimeDecode;
    }

    void InitModel(std::string encoder_model, std::string decoder_model, std::string inpaint_model)
    {
        mSam.Load(encoder_model, decoder_model);
        //        mInpaint.Load(inpaint_model);

        if (string_utility<std::string>::ends_with(inpaint_model, ".onnx"))
        {
            mInpaint.reset(new LamaInpaintOnnx);
        }
        else if (string_utility<std::string>::ends_with(inpaint_model, ".axmodel"))
        {
            mInpaint.reset(new LamaInpaintAX650);
        }
        else
        {
            fprintf(stderr, "no impl for %s\n", inpaint_model.c_str());
        }
        mInpaint->Load(inpaint_model);
    }

    void ShowRemoveObject(int dilate_size, QProgressBar *bar, bool remove_mask_by_merge = true)
    {
        if (!cur_image.bits() || !grab_masks.size())
        {
            return;
        }
        int channel = cur_image.format() == QImage::Format_BGR888 ? 3 : 4;
        int stride = cur_image.bytesPerLine();
        cv::Mat src(cur_image.height(), cur_image.width(), CV_8UC(channel), cur_image.bits(), stride);
        cv::Mat rgb;
        if (channel == 3)
            src.copyTo(rgb);
        else if (channel == 4)
            cv::cvtColor(src, rgb, cv::COLOR_RGBA2RGB);

        pt_img_first = QPoint(-10000, -10000);
        pt_img_secend = QPoint(-10000, -10000);
        cv::Mat inpainted = rgb;
        if (bar)
        {
            bar->setValue(0);
            bar->setMinimum(0);
            bar->setMaximum(grab_masks.size());
        }
        if (remove_mask_by_merge)
        {
            if (grab_masks.size())
            {
                auto base_mask = grab_masks[0];
                if (bar)
                    bar->setValue(bar->value() + 1);

                // merge all mask
                for (size_t i = 1; i < grab_masks.size(); i++)
                {
                    base_mask |= grab_masks[i];
                    if (bar)
                        bar->setValue(bar->value() + 1);
                }

                auto time_start = std::chrono::high_resolution_clock::now();
                inpainted = mInpaint->Inpaint(inpainted, base_mask, dilate_size);
                auto time_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = time_end - time_start;
                std::cout << "Inpaint Inference Cost time : " << diff.count() << "s" << std::endl;
                QImage qinpainted(inpainted.data, inpainted.cols, inpainted.rows, inpainted.step1(), QImage::Format_BGR888);
                cur_image = qinpainted.copy();
                cur_masks.clear();
                repaint();
            }
        }
        else
        {
            for (auto grab_mask : grab_masks)
            {
                auto time_start = std::chrono::high_resolution_clock::now();
                inpainted = mInpaint->Inpaint(inpainted, grab_mask, dilate_size);
                auto time_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = time_end - time_start;
                std::cout << "Inpaint Inference Cost time : " << diff.count() << "s" << std::endl;
                QImage qinpainted(inpainted.data, inpainted.cols, inpainted.rows, inpainted.step1(), QImage::Format_BGR888);
                cur_image = qinpainted.copy();
                if (cur_masks.size())
                    cur_masks.removeFirst();
                repaint();
                if (bar)
                    bar->setValue(bar->value() + 1);
            }
        }

        cur_masks.clear();
        rgba_masks.clear();
        grab_masks.clear();
        mSam.Encode(inpainted);
    }

    void Reset()
    {
        cur_image = bak_image.copy();
        SetImage(cur_image);
    }

    void RemoveLastMask()
    {
        if (cur_masks.size())
            cur_masks.removeLast();
        if (rgba_masks.size())
            rgba_masks.removeLast();
        if (grab_masks.size())
            grab_masks.removeLast();
    }

    static QPoint getSourcePoint(QSize window, QSize img, QPoint pt)
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

    static QPoint getWindowPoint(QSize window, QSize img, QPoint pt)
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
};
#endif // MYQLABEL_H
