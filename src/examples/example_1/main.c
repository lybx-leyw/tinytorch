#include "tensor.h"
#include "nn.h"
#include "autograd.h"
float model_run(Tensor* input,int lable,float lr,Operation* model);
Operation* create_model(LinearLayer* fc1,LinearLayer* fc2);

int main()
{
    clock_t time0 = clock();
    int shape[] = {1,9};
    int dim = 2;
    LinearLayer* fc1 = linear_create(9,16);
    LinearLayer* fc2 = linear_create(16,4);
    Operation* model = create_model(fc1,fc2);

    // 打开文件
    for (int epoch=0; epoch<10; epoch++) {
        FILE* fp = fopen("tictactoe.txt","r");
        if (!fp) {
            printf("file open error\n");
            exit(0);
        }

        char line[1000];
        int cnt=0;
        while (fgets(line,sizeof(line),fp)) {
            cnt++;
            float data[9];
            for (int i=0; i<9; i++) {
                switch (line[i]) {
                    case 'X':data[i]=1;break;
                    case 'O':data[i]=2;break;
                    default:data[i]=0;break;
                }
            }
            Tensor* input = tensor_create(data,shape,dim);
            int label;
            switch (line[10]) {
                case 'P':label=0;break;
                case 'N':label=1;break;
                case 'M':label=2;break;
                case 'Q':label=3;break;
                default: 
                    printf("label error");
                    return 1;
            }
            float loss=model_run(input,label,0.001,model);
            if (cnt%10==0) {
                FILE* tp = fopen("loss_c.txt","a");
                fprintf(tp,"loss:%10.6f\n",100000*loss);
                fclose(tp);
            }
        }
        fclose(fp);
    }
    clock_t time1 = clock();
    printf("%ds",(time1-time0)/CLOCKS_PER_SEC);
    return 0;
}