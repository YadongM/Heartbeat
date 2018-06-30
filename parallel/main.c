#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"
#include "read_npy.h"
#include "string.h"
#include <time.h>
#include "mpi.h"

#define SHOW_TIME 1
int main(void){
    //读取数据
    int id,nb_procs;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&id);
    MPI_Comm_size(MPI_COMM_WORLD,&nb_procs);
#if SHOW_TIME==1
	int endcount=0;
	double begin,end;
	if(id==0){
		begin=MPI_Wtime();
	}
#endif
    struct data_box *train_data=npy_load("X_train.npy",id,nb_procs);
    struct data_box *train_label=npy_load("y_train.npy",id,nb_procs);
    struct data_box *test_data=npy_load("X_test.npy",id,nb_procs);
    struct data_box *test_label=npy_load("y_test.npy",id,nb_procs);
   //定义一个用于喂数据的结构体
    struct feed_data feed_box;
    //定义迭代的次数和每一个batch包含的数据个数
    int nb_epoch=50;
    int batch_size=1000;
    batch_size/=nb_procs;
    //定义一个用于存放条件的结构体
    struct data_box con;
    int sample_size=1,classes;
    con.ndims=train_data->ndims;
    con.shape=(int*)malloc(con.ndims*sizeof(int));
    con.shape[0]=batch_size;
    for(int i=1;i<con.ndims;i++){
        con.shape[i]=train_data->shape[i];
        sample_size*=con.shape[i];
    }
    classes=train_label->shape[1];
    //使用上面定义的结构体初始化神经网络
    struct CNN *cnn=cnn_init(&con);
    int weight_size=cnn->weight_size;
    double *w,*dw,*m,*v,*buf,*loop_result;
    w=(double *)malloc(weight_size*sizeof(double));
    buf=(double *)calloc(weight_size,sizeof(double));
    if(id==0){
        dw=(double *)malloc(weight_size*sizeof(double));
        m=(double *)calloc(weight_size,sizeof(double));
        v=(double *)calloc(weight_size,sizeof(double));
        loop_result=(double *)malloc(2*sizeof(double));
    }

    int loop_time,the_last_time;
    double loss,acc;
	double acc_flag;
    double *result;

    if(id==0){
        pack_weight(cnn,w);
    }
    MPI_Bcast(w,weight_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
    load_weight(cnn,w);

    for(int i=0;i<nb_epoch;i++){
        //train
        loop_time=train_data->shape[0]/batch_size;
        the_last_time=train_data->shape[0]%batch_size;
        if(the_last_time>0) loop_time++;
        loss=acc=0;

        for(int j=0;j<loop_time;j++){
            memset(buf,0,weight_size*sizeof(double));

            feed_box.data=train_data->data+j*batch_size*sample_size;
            feed_box.label=train_label->data+j*batch_size*classes;
            if(j==loop_time-1&&the_last_time!=0){
                feed_box.sample_num=the_last_time;
            }else{
                feed_box.sample_num=batch_size;
            }
            //喂数据
            feed(cnn,&feed_box);
            //运行神经网络
            result=go(cnn,TRAIN);
            //计算平均loss和acc并输出
            MPI_Reduce(result,loop_result,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

			acc_flag = loop_result[1]/nb_procs; 	
            MPI_Bcast(&acc_flag,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
            if(id==0){
                printf("train : epoch:%d,loop:%d,loss:%f,acc:%f\n",i,j,loop_result[0]/nb_procs,acc_flag);
            }
            free(result);
#if SHOW_TIME==1
			if(acc_flag>=0.9){
				endcount++;
			 	if(id==0&&endcount==5){
					end=MPI_Wtime();
					printf("***************\n%d process:%fs\n***************\n", nb_procs, end-begin);
				}  
				if(endcount==5){
					exit(0);
				}
			}
#endif

            pack_dweight(cnn,buf);
				//begin=MPI_Wtime();
            MPI_Reduce(buf,dw,weight_size,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
				//end=MPI_Wtime();
            if(id==0){
				printf("%f",cnn->adam_para.t);
                adam(cnn,w,dw,m,v,weight_size);
		
				//printf("%fs\n",end-begin);
            }
            MPI_Bcast(w,weight_size,MPI_DOUBLE,0,MPI_COMM_WORLD);
            load_weight(cnn,w);

        }

        //test
        loop_time=test_data->shape[0]/batch_size;
        the_last_time=test_data->shape[0]%batch_size;
        if(the_last_time>0) loop_time++;
        loss=acc=0;
        for(int j=0;j<loop_time;j++){
            feed_box.data=test_data->data+j*sample_size;
            feed_box.label=test_label->data+j*classes;
            if(j==loop_time-1&&the_last_time!=0){
                feed_box.sample_num=the_last_time;
            }else{
                feed_box.sample_num=batch_size;
            }
            feed(cnn,&feed_box);
            result=go(cnn,TEST);
            loss=loss*j/(j+1)+result[0]/(j+1);
            acc=acc*j/(j+1)+result[1]/(j+1);
            free(result);
            printf("test : loss:%f,acc:%f\n",loss,acc);
        }
    }
    MPI_Finalize();
    return 0;
}
