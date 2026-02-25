#include "nn.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)))

static int calculate_output_size(int input_size, int kernel_size, int padding, int stride) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}
ConvLayer* conv_create(int in_channels, int out_channels, int kernel_size, int stride, int padding)
{
    ConvLayer* ret_layer = Calloc(1, ConvLayer);

    // [out_channels, in_channels, kernel_size, kernel_size]
    int weight_shape[] = {out_channels, in_channels, kernel_size, kernel_size};

    ret_layer->weight = tensor_rand(weight_shape, 4, -1, 1);
    ret_layer->kernel_size = kernel_size;
    ret_layer->stride = stride;
    ret_layer->padding = padding;

    return ret_layer;
}

Tensor* conv_forward(ConvLayer* layer, Tensor* input)
{
    int batch = input->shape[0];
    int in_height = input->shape[1];
    int in_width = input->shape[2];
    int in_channels = input->shape[3];

    int kernel_size = layer->kernel_size;
    int stride = layer->stride;
    int padding = layer->padding;

    int out_height = calculate_output_size(in_height, kernel_size, padding, stride);
    int out_width = calculate_output_size(in_width, kernel_size, padding, stride);
    int out_channels = layer->weight->shape[0];

    int output_shape[] = {batch, out_height, out_width, out_channels};
    Tensor* output = tensor_zeros(output_shape, 4);

    // 初始化或更新梯度
    if (!layer->weight->grad) {
        int weight_grad_shape[] = {out_channels, in_channels, kernel_size, kernel_size};
        layer->weight->grad = tensor_zeros(weight_grad_shape, 4);
    }

    if (!input->grad) {
        int input_grad_shape[] = {batch, in_height, in_width, in_channels};
        input->grad = tensor_zeros(input_grad_shape, 4);
    }

    // 执行卷积操作
    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                for (int oc = 0; oc < out_channels; oc++) {
                    int ih_start = oh * stride - padding;
                    int iw_start = ow * stride - padding;

                    float sum = 0.0f;

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = ih_start + kh;
                                int iw = iw_start + kw;

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + ic;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                                    sum += input->data[input_idx] * layer->weight->data[weight_idx];
                                }
                            }
                        }
                    }

                    int output_idx = ((b * out_height + oh) * out_width + ow) * out_channels + oc;
                    output->data[output_idx] = sum;
                }
            }
        }
    }

    return output;
}

Tensor* conv_backward(ConvLayer* layer, Tensor* input, Tensor* upstream_grad)
{
    int batch = input->shape[0];
    int in_height = input->shape[1];
    int in_width = input->shape[2];
    int in_channels = input->shape[3];

    int kernel_size = layer->kernel_size;
    int stride = layer->stride;
    int padding = layer->padding;

    int out_height = upstream_grad->shape[1];
    int out_width = upstream_grad->shape[2];
    int out_channels = upstream_grad->shape[3];

    // 重置梯度为0
    for (int i = 0; i < layer->weight->grad->size; i++) {
        layer->weight->grad->data[i] = 0.0f;
    }
    for (int i = 0; i < input->grad->size; i++) {
        input->grad->data[i] = 0.0f;
    }

    // 计算权重梯度和输入梯度
    for (int b = 0; b < batch; b++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                for (int oc = 0; oc < out_channels; oc++) {
                    int ih_start = oh * stride - padding;
                    int iw_start = ow * stride - padding;

                    int grad_idx = ((b * out_height + oh) * out_width + ow) * out_channels + oc;
                    float grad_value = upstream_grad->data[grad_idx];

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_size; kh++) {
                            for (int kw = 0; kw < kernel_size; kw++) {
                                int ih = ih_start + kh;
                                int iw = iw_start + kw;

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int input_idx = ((b * in_height + ih) * in_width + iw) * in_channels + ic;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                                    // 权重梯度
                                    layer->weight->grad->data[weight_idx] += input->data[input_idx] * grad_value;

                                    // 输入梯度
                                    input->grad->data[input_idx] += layer->weight->data[weight_idx] * grad_value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return input->grad;
}

void conv_sgd_update(ConvLayer* layer, float lr)
{
    if (layer->weight->requires_grad && layer->weight->grad) {
        for (int i = 0; i < layer->weight->size; i++) {
            layer->weight->data[i] -= lr * layer->weight->grad->data[i];
        }
    }
}

void conv_free(ConvLayer* layer)
{
    if (layer == NULL) return;
    if (layer->weight) tensor_free(layer->weight);
    if (layer->weight && layer->weight->grad) tensor_free(layer->weight->grad);
    free(layer);
}
