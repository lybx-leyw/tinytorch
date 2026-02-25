#include "tensor.h"
#include "nn.h"
#include "autograd.h"
#include "optim.h"
#include <time.h>

Operation* create_conv_model(ConvLayer* conv,LinearLayer* fc);
float model_run(Tensor* input,int label,float lr,Operation* model);

int main()
{
    clock_t time0 = clock();

    ConvLayer* conv = conv_create(1, 4, 3, 1, 0);
    LinearLayer* fc = linear_create(4, 4);
    Operation* model = create_conv_model(conv, fc);

    for (int epoch = 0; epoch < 10; epoch++) {
        FILE* fp = fopen("tictactoe.txt","r");
        if (!fp) {
            printf("file open error\n");
            exit(0);
        }

        char line[1000];
        int cnt = 0;
        float total_loss = 0.0f;

        while (fgets(line,sizeof(line),fp)) {
            cnt++;
            float data[9];
            for (int i = 0; i < 9; i++) {
                switch (line[i]) {
                    case 'X':data[i] = 1.0f;break;
                    case 'O':data[i] = 2.0f;break;
                    default:data[i] = 0.0f;break;
                }
            }
            int shape[] = {1, 3, 3, 1};
            Tensor* input = tensor_create(data, shape, 4);
            int label;
            switch (line[10]) {
                case 'P':label = 0;break;
                case 'N':label = 1;break;
                case 'M':label = 2;break;
                case 'Q':label = 3;break;
                default:
                    printf("label error");
                    return 1;
            }
            float loss = model_run(input, label, 0.001, model);
            total_loss += loss;
            tensor_free(input);
            if (cnt % 10 == 0) {
                FILE* tp = fopen("loss_conv.txt","a");
                fprintf(tp,"loss:%10.6f\n",100000*total_loss/10);
                fclose(tp);
                total_loss = 0.0f;
            }
        }
        fclose(fp);
        printf("Epoch %d completed\n", epoch + 1);
    }
    clock_t time1 = clock();
    printf("Training time: %ds\n",(time1-time0)/CLOCKS_PER_SEC);
    return 0;
}

