#pragma once
#include "vector"
#include "string"

typedef enum _color_space_e
{
    axdl_color_space_unknown,
    axdl_color_space_nv12,
    axdl_color_space_nv21,
    axdl_color_space_bgr,
    axdl_color_space_rgb,
} ax_color_space_e;

typedef struct _image_t
{
    unsigned long long int pPhy;
    void *pVir;
    unsigned int nSize;
    unsigned int nWidth;
    unsigned int nHeight;
    ax_color_space_e eDtype;
    union
    {
        int tStride_H, tStride_W, tStride_C;
    };
} ax_image_t;

typedef struct
{
    std::string sName;
    unsigned int nIdx;
    std::vector<unsigned int> vShape;
    int nSize;
    unsigned long phyAddr;
    void *pVirAddr;
} ax_runner_tensor_t;

class ax_runner_base
{
protected:
    std::vector<ax_runner_tensor_t> mtensors;
    std::vector<ax_runner_tensor_t> minput_tensors;

public:
    virtual int init(const char *model_file) = 0;

    virtual void deinit() = 0;

    int get_num_outputs() { return mtensors.size(); };

    const ax_runner_tensor_t &get_input(int idx) { return minput_tensors[idx]; }
    const ax_runner_tensor_t *get_inputs_ptr() { return minput_tensors.data(); }

    const ax_runner_tensor_t &get_output(int idx) { return mtensors[idx]; }
    const ax_runner_tensor_t *get_outputs_ptr() { return mtensors.data(); }

    virtual int get_algo_width() = 0;
    virtual int get_algo_height() = 0;
    virtual ax_color_space_e get_color_space() = 0;

    virtual int inference(ax_image_t *pstFrame) = 0;
    virtual int inference() = 0;

    int operator()(ax_image_t *pstFrame)
    {
        return inference(pstFrame);
    }
};
