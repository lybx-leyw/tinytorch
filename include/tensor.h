#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef enum{LINEAR_LAYER,RELU_ACTIVE_LAYER,CONV_LAYER,CROSS_ENTROY_LOSS_LAYER} LAYERTYPE;
typedef struct _tensor
{
    float* data;
    int* shape;
    int ndim;
    int size;
    int requires_grad;
    struct _tensor* grad;    
} Tensor;

Tensor* tensor_create(float* data, int* shape, int ndim);
Tensor* tensor_zeros(int* shape, int ndim);
Tensor* tensor_ones(int* shape, int ndim);
Tensor* tensor_rand(int* shape,int ndim,int min,int max);
void tensor_free(Tensor* t);
void tensor_print(Tensor* t);

void tensor_squeeze(Tensor* t,int dim);
void tensor_unsqueeze(Tensor* t,int dim);
Tensor* tensor_neg(Tensor* t);
Tensor* tensor_exp(Tensor* t);
Tensor* tensor_log(Tensor* t);
Tensor* tensor_relu(Tensor* t);
Tensor* tensor_sigmoid(Tensor* t);
Tensor* tensor_countdown(Tensor* t);
Tensor* tensor_relu_index(Tensor* t);

Tensor* tensor_add(Tensor* t1,Tensor* t2);
Tensor* tensor_sub(Tensor* t1,Tensor* t2);
Tensor* tensor_mul(Tensor* t1,Tensor* t2);
Tensor* tensor_scalar_mul(Tensor* t1,float number);
Tensor* tensor_divide(Tensor* t1,Tensor* t2);
Tensor* tensor_matmul(Tensor* t1,Tensor* t2);
Tensor* tensor_cat(Tensor** tp,int nt,int dim);
void tensor_permute(Tensor* t,int dim1,int dim2);
void tensor_transpose(Tensor* t);

void print_shape(Tensor* t);
Tensor* tensor_sum(Tensor* t);
Tensor* tensor_mean(Tensor* t);

Tensor* tensor_repeat(Tensor* t,int dim,int multiple);
Tensor* tensor_clone(Tensor* t);
Tensor* tensor_softmax(Tensor* t);

#endif