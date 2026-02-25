#ifndef __OPTIM_H__
#define __OPTIM_H__
#include "nn.h"
#include "autograd.h"

// SGD随机梯度下降法
void sgd_step(LAYERTYPE layer_type,void* update_layer,float lr);
void model_step(Operation* model,float lr);

#endif