#include "optim.h"

void sgd_step(LAYERTYPE layer_type,void* update_layer,
    float lr)
{
    switch (layer_type) {
        case LINEAR_LAYER:
            linear_sgd_update((LinearLayer*)update_layer,lr);break;
        case CONV_LAYER:
            conv_sgd_update((ConvLayer*)update_layer,lr);break;
    }
}
void model_step(Operation* model,float lr)
{
    while (model) {
        sgd_step(model->layer_type,model->params,lr);
        model = model->next_op;
    }
}