#include "tensor.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)))

// 张量初始化
static int cacul_size_op(int* shape,int ndim)
{
    int size = 1;
    for (int i=0; i<ndim; i++) {
        size *= shape[i];
    }
    return size;
}
static Tensor* tensor_assignment_op(float* data,int* shape,
    int ndim,int number)
{
    Tensor* retTensor = Calloc(1,Tensor);
    retTensor->ndim = ndim;
    retTensor->shape = Calloc(ndim,int);
    // 计算size
    int size = cacul_size_op(shape,ndim);
    // 拷贝shape
    for (int i=0; i<ndim; i++) {
        retTensor->shape[i] = shape[i];
    }
    retTensor->size = size;

    retTensor->data = Calloc(size,float);
    if (!data && number==0) return retTensor;
    for (int i=0; i<size; i++) {
        retTensor->data[i] = data?data[i]:number;
    }

    retTensor->requires_grad = 1;
    return retTensor;
}
Tensor* tensor_create(float* data,int* shape,int ndim)
{
   return tensor_assignment_op(data,shape,ndim,0);
}
Tensor* tensor_zeros(int* shape,int ndim)
{
    return tensor_assignment_op(NULL,shape,ndim,0);
}
Tensor* tensor_ones(int* shape,int ndim)
{
    return tensor_assignment_op(NULL,shape,ndim,1);
}
Tensor* tensor_rand(int* shape,int ndim,int min,int max)
{
    srand(time(0));
    int size = cacul_size_op(shape,ndim);
    float* data = Calloc(size,float);
    for (int i=0; i<size; i++) {
        data[i] = (0.000001*(rand()%1000000))+rand()%(max-min)+min;
    }
    Tensor* ret = tensor_assignment_op(data,shape,ndim,0);   
    free(data);
    return ret;
}

// 张量辅助工具
void tensor_free(Tensor* t)
{
    if (t==NULL) return;
    free(t->data);
    free(t->shape);
    free(t);
}
static void print_op(Tensor* t) 
{
    // 递归实现
    if (t->ndim==1) {
        printf("[");
        for(int i=0; i<t->shape[0]; i++) {
            printf(" %.6f ",t->data[i]);
        }
        printf("]");
        return;
    } else {
        printf("[");
        // 按照shape[0]的维度划分数据分为很多小的张量
        int* new_shape = t->shape+1;
        float* cur_date = t->data; 
        int new_dim = t->ndim-1;
        for(int i=0; i<t->shape[0]; i++) {
            Tensor* t_tensor = tensor_create(cur_date,new_shape,new_dim);
            print_op(t_tensor);
            cur_date += t_tensor->size;
            tensor_free(t_tensor);
        }
        printf("]");
    }
}
void tensor_print(Tensor* t)
{
    print_op(t);
    printf("\n");
}

// 张量形状调整工具
void tensor_squeeze(Tensor* t,int dim)
{
    if (dim>=t->ndim || dim<0) {
        return;
    }
    for (int i=dim; i<t->ndim-1; i++) {
        t->shape[i] = t->shape[i+1];
    }
    t->ndim = t->ndim-1;
    t->shape = (int*)realloc(t->shape,t->ndim*sizeof(int));
}
void tensor_unsqueeze(Tensor* t,int dim)
{
    if (dim>t->ndim || dim<0) {
        return;
    }
    t->ndim = t->ndim+1;
    t->shape = (int*)realloc(t->shape,t->ndim*sizeof(int));
    for (int i=t->ndim-1; i>dim; i--) {
        t->shape[i] = t->shape[i-1];
    }
    t->shape[dim]=1;
}

// 单目运算
static float max(float a,float b) {
    return (a > b) ? a : b;
}
typedef enum {NEG,EXP,LOG,RELU,SIGMOID,DOWN,RELU_INDEX} MonoType;
static Tensor* mono_tensor_op(Tensor* t,MonoType type)
{
    int size = t->size;
    float* data = Calloc(size,float);
    for (int i=0; i<size; i++) {
        switch (type) {
            case NEG:data[i] = -t->data[i];break;
            case EXP:data[i] = exp(t->data[i]);break;
            case LOG:data[i] = log(t->data[i]);break;
            case RELU:data[i] = max(0.0,t->data[i]);break;
            case SIGMOID: {
                float x = t->data[i];
                if (t->data[i] >= 0) {
                    data[i] = 1.0f / (1.0f + expf(-x));
                } else {
                    float ex = expf(x);
                    data[i] = ex / (1.0f + ex);
                }
                break;
            }
            case DOWN:data[i] = 1.0/t->data[i];break;
            case RELU_INDEX:data[i]=t->data[i]>0?1:0;break;
            default:return NULL;
        }
    }
    Tensor* ret = tensor_assignment_op(data,t->shape,t->ndim,0);
    free(data);
    return ret;
}
Tensor* tensor_neg(Tensor* t)
{
    return mono_tensor_op(t,NEG);
}
Tensor* tensor_exp(Tensor* t)
{
    return mono_tensor_op(t,EXP);
}
Tensor* tensor_log(Tensor* t)
{
    return mono_tensor_op(t,LOG);
}
Tensor* tensor_relu(Tensor* t)
{
    return mono_tensor_op(t,RELU);
}
Tensor* tensor_sigmoid(Tensor* t)
{
    return mono_tensor_op(t,SIGMOID);
}
Tensor* tensor_countdown(Tensor* t)
{
    return mono_tensor_op(t,DOWN);
}
Tensor* tensor_relu_index(Tensor* t)
{
    return mono_tensor_op(t,RELU_INDEX); 
}

//双目运算
typedef enum {ADD,SUB,MUL,SCAMUL,DIVIDE} BinoType;
static Tensor* bino_tensor_op(Tensor* t1,Tensor* t2,BinoType type,float number)
{
    if (!t2) {
        type = SCAMUL;
    } else if (t1->size%t2->size) {
        return NULL;
    }
    int size = t1->size;
    float* data = Calloc(size,float);
    for (int i=0; i<size; i++) {
        int j;
        if (t2) j=i%(t2->size);
        switch (type) {
            case ADD:data[i]=t1->data[i]+t2->data[j];break;
            case SUB:data[i]=t1->data[i]-t2->data[j];break;
            case MUL:data[i]=t1->data[i]*t2->data[j];break;
            case SCAMUL:data[i]=t1->data[i]*number;break;
            case DIVIDE:data[i]=t1->data[i]/t2->data[j];break;
            default:return NULL;
        }
    }
    Tensor* ret = tensor_assignment_op(data,t1->shape,t1->ndim,0);
    free(data);
    return ret;
}
Tensor* tensor_add(Tensor* t1,Tensor* t2)
{
    return bino_tensor_op(t1,t2,ADD,0);
}
Tensor* tensor_sub(Tensor* t1,Tensor* t2)
{
    return bino_tensor_op(t1,t2,SUB,0);
}
Tensor* tensor_mul(Tensor* t1,Tensor* t2)
{
    return bino_tensor_op(t1,t2,MUL,0);
}
Tensor* tensor_scalar_mul(Tensor* t1,float number)
{
    return bino_tensor_op(t1,NULL,SCAMUL,number);
}
Tensor* tensor_divide(Tensor* t1,Tensor* t2)
{
    return bino_tensor_op(t1,t2,DIVIDE,0);
}

Tensor* tensor_matmul(Tensor* t1,Tensor* t2)
{
    if(t1->shape[t1->ndim-1]!=t2->shape[t2->ndim-2]) {
        return NULL;
    }
    int dim_2 = t2->shape[t2->ndim-1];
    int same_d = t2->shape[t2->ndim-2];
    int dim_1 = t1->size/same_d;
    
    int new_size = dim_1*dim_2;
    float* ret_data = Calloc(new_size,float);
    for (int i=0; i<dim_1; i++) {
        for (int j=0; j<dim_2; j++) {
            for (int z=0; z<same_d; z++) {
                ret_data[i*dim_2+j]+=t1->data[i*same_d+z]*t2->data[z*dim_2+j];    
            }
        }
    }

    int* ret_shape = Calloc(t1->ndim,int);
    for (int i=0; i<t1->ndim-2; i++) {
        ret_shape[i] = t1->shape[i];
    }
    ret_shape[t1->ndim-2] = t1->shape[t1->ndim-2];
    ret_shape[t1->ndim-1] = dim_2;
    Tensor* ret = Calloc(1,Tensor);
    ret->data = ret_data;
    ret->ndim = t1->ndim;
    ret->shape = ret_shape;
    ret->size = new_size;
    return ret;
}

// 张量形状变换
Tensor* tensor_cat(Tensor** tp,int nt,int dim)
{   
    // 按顺序合并所有数据
    int ret_size = 0;
    for (int i=0; i<nt; i++) {
        ret_size += tp[i]->size;
    }
    float* ret_data = Calloc(ret_size,float);
    int index = 0;
    int last_dim_idx = tp[0]->ndim-1;
    if (dim!=last_dim_idx) {
        for (int i=0; i<nt; i++) {
            for (int j=0; j<tp[i]->size; j++) {
                ret_data[index] = tp[i]->data[j];
                index++;
            }
        }
    } else {
        int number=tp[0]->size/tp[0]->shape[last_dim_idx];
        for (int idx=0; idx<number; idx++) {
            for (int i=0; i<nt; i++) {
                int d_size=tp[i]->shape[last_dim_idx];
                for (int j=d_size*idx; j<d_size*(idx+1); j++) {
                    ret_data[index] = tp[i]->data[j];
                    index++;
                }
            }
        }
    }

    // 合并形状
    int* ret_shape = Calloc(tp[0]->ndim,int);
    for (int i = 0; i<tp[0]->ndim; i++) {
        if (i==dim) continue;
        ret_shape[i] = tp[0]->shape[i];
    }
    for (int i=0; i<nt; i++) {
        ret_shape[dim] += tp[i]->shape[dim];
    }
    Tensor* ret = tensor_assignment_op(ret_data,ret_shape,tp[0]->ndim,0);
    free(ret_data);
    free(ret_shape);
    return ret;
}

#define SWAP(a,b,type) {type temp=a;a=b;b=temp;}
void tensor_permute(Tensor* t,int dim1,int dim2)
{
    if (dim1==dim2) {
        return;
    }
    if (t->ndim-1!=dim2) {
        SWAP(t->shape[dim1],t->shape[dim2],int);
    } else {
        tensor_transpose(t);
        tensor_permute(t,dim1,dim2-1);
    }
}
void tensor_transpose(Tensor* t)
{
    int dim1 = t->shape[t->ndim-2];
    int dim2 = t->shape[t->ndim-1];
    int number = t->size/dim1/dim2;
    SWAP(t->shape[t->ndim-2],t->shape[t->ndim-1],int);
    
    for (int index=0; index<number; index++) {
        for (int i=0; i<dim1; i++) {
            for (int j=0; j<dim2; j++) {
                SWAP(t->data[index*dim1*dim2+j*dim1+i],
                    t->data[index*dim1*dim2+i*dim2+j],float);
            }
        }
    }
}

// 形状辅助
void print_shape(Tensor* t)
{
    printf("tensor[");
    for (int i=0; i<t->ndim-1; i++) {
        printf(" %d ,",t->shape[i]);
    }
    printf(" %d ]\n",t->shape[t->ndim-1]);
}

// 归约运算
static Tensor* tensor_sum_op(Tensor* t,float scale) 
{
    int d_size = t->shape[t->ndim-1];
    int number=t->size/d_size;
    int* new_shape = Calloc(t->ndim,int);
    for (int i=0; i<t->ndim-1; i++) {
        new_shape[i] = t->shape[i];
    }
    new_shape[t->ndim-1] = 1;

    float* new_data = Calloc(number,float);
    for (int index=0; index<number; index++) {
        float sum=0.0;
        for (int i=index*d_size; i<(index+1)*d_size; i++) {
            sum += t->data[i];
        }
        new_data[index]=sum*scale;
    }

    Tensor* ret = Calloc(1,Tensor);
    ret->data = new_data;
    ret->shape = new_shape;
    ret->ndim = t->ndim;
    ret->size = number;
    return ret;
}
Tensor* tensor_sum(Tensor* t)
{
    return tensor_sum_op(t,1);
}
Tensor* tensor_mean(Tensor* t)
{
    int d_size = t->shape[t->ndim-1];
    float scale = 1.0/d_size;
    return tensor_sum_op(t,scale);
}

// 其他
Tensor* tensor_repeat(Tensor* t,int dim,int multiple)
{
    Tensor* all_t[multiple];
    for (int i=0; i<multiple; i++) {
        all_t[i]=t;
    }
    return tensor_cat(all_t,multiple,dim);
}
Tensor* tensor_clone(Tensor* t)
{
    return tensor_assignment_op(t->data,t->shape,t->ndim,0);
}

static float get_max_tensor_data(Tensor* t)
{
    float max = -10000;
    for (int i=0; i<t->size; i++) {
        if (t->data[i] > max) {
            max = t->data[i];
        }
    }
    return max;
}
Tensor* tensor_softmax(Tensor* t)
{
    float max = get_max_tensor_data(t);
    int shape = 1;
    Tensor* t_max = tensor_create(&max,&shape,1);
    Tensor* t_input = tensor_sub(t,t_max);

    Tensor* exp = tensor_exp(t_input);
    Tensor* sum = tensor_sum(exp);
    Tensor* repeat_sum = tensor_repeat(sum,sum->ndim-1,exp->shape[exp->ndim-1]);
    Tensor* ret = tensor_divide(exp,repeat_sum);
    
    tensor_free(t_max);
    tensor_free(t_input); 
    tensor_free(exp);
    tensor_free(sum);
    tensor_free(repeat_sum);        
    return ret;
}