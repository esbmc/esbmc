#include <pthread.h>
#include <global_vars.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void *DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv(void *arg)
{
int this_addr;
    int iftmp_2e_546;
    int tmp__1;
    int alloca_20_point;
    int tmp__2;
    int tmp__3;
    int tmp__4;
    int tmp__5;
    int tmp__6;
    int tmp__7;
    int tmp__8;
    int tmp__9;
    int this;

    ///+ code for struct used as input/output for pthreads
    struct inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv *my_struct;
    my_struct = (struct inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv *) arg;
    alloca_20_point = 0;
    this_addr = this;
    tmp__2 = this_addr;
    tmp__3 = /*ROBREAD DUT_H_N4_0xbfd39db4 0x9e147a0*/ my_struct->input_0;
    tmp__4 = tmp__3;
    if((((((tmp__4 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb;


bb:
    tmp__5 = this_addr;
    tmp__6 = /*ROBREAD DUT_H_N4_0xbfd39dec 0x9eaf678*/ my_struct->input_1;
    tmp__7 = tmp__6;
    if((((((tmp__7 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb9;


bb8:
    iftmp_2e_546 = ((int )1);
    goto bb10;


bb9:
    iftmp_2e_546 = ((int )0);
    goto bb10;


bb10:
    tmp__8 = iftmp_2e_546;
    tmp__1 = tmp__8;
    tmp__9 = this_addr;
    /*ROBWRITE DUT_H_N4_0xbfd39e24 0x9f2df10*/
    my_struct->output_0 = tmp__8;
;
    DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv_join = 1;
    pthread_exit(NULL);
}


void *DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv(void *arg)
{
int this_addr;
    int iftmp_2e_546;
    int tmp__1;
    int alloca_20_point;
    int tmp__2;
    int tmp__3;
    int tmp__4;
    int tmp__5;
    int tmp__6;
    int tmp__7;
    int tmp__8;
    int tmp__9;
    int this;

    ///+ code for struct used as input/output for pthreads
    struct inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv *my_struct;
    my_struct = (struct inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv *) arg;
    alloca_20_point = 0;
    this_addr = this;
    tmp__2 = this_addr;
    tmp__3 = /*ROBREAD DUT_H_N3_0xbfd39cac 0x9e4dfc0*/ my_struct->input_0;
    tmp__4 = tmp__3;
    if((((((tmp__4 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb;


bb:
    tmp__5 = this_addr;
    tmp__6 = /*ROBREAD DUT_H_N3_0xbfd39ce4 0x9e4ffb0*/ my_struct->input_1;
    tmp__7 = tmp__6;
    if((((((tmp__7 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb9;


bb8:
    iftmp_2e_546 = ((int )1);
    goto bb10;


bb9:
    iftmp_2e_546 = ((int )0);
    goto bb10;


bb10:
    tmp__8 = iftmp_2e_546;
    tmp__1 = tmp__8;
    tmp__9 = this_addr;
    /*ROBWRITE DUT_H_N3_0xbfd39d1c 0x9f2d310*/
    my_struct->output_0 = tmp__8;
;
    DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv_join = 1;
    pthread_exit(NULL);
}


void *DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv(void *arg)
{
int this_addr;
    int iftmp_2e_546;
    int tmp__1;
    int alloca_20_point;
    int tmp__2;
    int tmp__3;
    int tmp__4;
    int tmp__5;
    int tmp__6;
    int tmp__7;
    int tmp__8;
    int tmp__9;
    int this;

    ///+ code for struct used as input/output for pthreads
    struct inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv *my_struct;
    my_struct = (struct inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv *) arg;
    alloca_20_point = 0;
    this_addr = this;
    tmp__2 = this_addr;
    tmp__3 = /*ROBREAD DUT_H_N2_0xbfd39ba4 0x9f1fed8*/ my_struct->input_0;
    tmp__4 = tmp__3;
    if((((((tmp__4 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb;


bb:
    tmp__5 = this_addr;
    tmp__6 = /*ROBREAD DUT_H_N2_0xbfd39bdc 0x9f22b58*/ my_struct->input_1;
    tmp__7 = tmp__6;
    if((((((tmp__7 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb9;


bb8:
    iftmp_2e_546 = ((int )1);
    goto bb10;


bb9:
    iftmp_2e_546 = ((int )0);
    goto bb10;


bb10:
    tmp__8 = iftmp_2e_546;
    tmp__1 = tmp__8;
    tmp__9 = this_addr;
    /*ROBWRITE DUT_H_N2_0xbfd39c14 0x9f257d8*/
    my_struct->output_0 = tmp__8;
;
    DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv_join = 1;
    pthread_exit(NULL);
}


void *DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv(void *arg)
{
int this_addr;
    int iftmp_2e_546;
    int tmp__1;
    int alloca_20_point;
    int tmp__2;
    int tmp__3;
    int tmp__4;
    int tmp__5;
    int tmp__6;
    int tmp__7;
    int tmp__8;
    int tmp__9;
    int this;

    ///+ code for struct used as input/output for pthreads
    struct inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv *my_struct;
    my_struct = (struct inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv *) arg;
    alloca_20_point = 0;
    this_addr = this;
    tmp__2 = this_addr;
    tmp__3 = /*ROBREAD DUT_H_N1_0xbfd39a9c 0x9e9d430*/ my_struct->input_0;
    tmp__4 = tmp__3;
    if((((((tmp__4 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb;


bb:
    tmp__5 = this_addr;
    tmp__6 = /*ROBREAD DUT_H_N1_0xbfd39ad4 0x9eadad8*/ my_struct->input_1;
    tmp__7 = tmp__6;
    if((((((tmp__7 != ((int )0)) ^ 1)&1))) != ((int )0))
      goto bb8;
    else
      goto bb9;


bb8:
    iftmp_2e_546 = ((int )1);
    goto bb10;


bb9:
    iftmp_2e_546 = ((int )0);
    goto bb10;


bb10:
    tmp__8 = iftmp_2e_546;
    tmp__1 = tmp__8;
    tmp__9 = this_addr;
    /*ROBWRITE DUT_H_N1_0xbfd39b0c 0x9e06438*/
    my_struct->output_0 = tmp__8;
;
    DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv_join = 1;
    pthread_exit(NULL);
}


void *DUT(void *arg)
{
    /* struct to be used for inputs and outputs */
    void *status;
    struct inputs_DUT *my_inputs;
    my_inputs = (struct inputs_DUT *) arg;
    my_inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv.input_0 = my_inputs->input_0;
    my_inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv.input_0 = my_inputs->input_0;
    my_inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv.input_1 = my_inputs->input_1;
    my_inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv.input_1 = my_inputs->input_1;

    /* create pthread */
    pthread_t id0;
    int t0=0;
    pthread_create(&id0, NULL, DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv, (void *)&my_inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv);

    if(DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv_join==1){
    my_inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv.input_1 = my_inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv.output_0;
    my_inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv.input_0 = my_inputs_DUT_H_N1__ZN4nand7do_nandEv_pid3_ZN4nand7do_nandEv.output_0;
    
    /* create pthread */
        pthread_t id1;
        int t1=1;
        pthread_create(&id1, NULL, DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv, (void *)&my_inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv);

        if(DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv_join==1){
        my_inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv.input_0 = my_inputs_DUT_H_N2__ZN4nand7do_nandEv_pid2_ZN4nand7do_nandEv.output_0;
        
    /* create pthread */
            pthread_t id2;
            int t2=2;
            pthread_create(&id2, NULL, DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv, (void *)&my_inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv);

            if(DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv_join==1){
            my_inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv.input_1 = my_inputs_DUT_H_N3__ZN4nand7do_nandEv_pid1_ZN4nand7do_nandEv.output_0;
            
    /* create pthread */
                pthread_t id3;
                int t3=3;
                pthread_create(&id3, NULL, DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv, (void *)&my_inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv);

                if(DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv_join==1){


                /* MODULE OUTPUTS */
                    my_inputs->output_0 = my_inputs_DUT_H_N4__ZN4nand7do_nandEv_pid0_ZN4nand7do_nandEv.output_0;
                    DUT_join = 1;
                    pthread_exit(NULL);
                }
            }
        }
    }
}

_Bool nondet_bool();

/**********************main C program***********************/
int main(){

   

    int rc;
    void *status;
    /* Connect module inputs to PI's */
    _Bool PI_0=nondet_bool();
    my_inputs_DUT.input_0 = PI_0;
    _Bool PI_1=nondet_bool();
    my_inputs_DUT.input_1 = PI_1;

    pthread_t id0;
    rc = pthread_create(&id0, NULL, DUT, &my_inputs_DUT);
    if (rc) {
        printf("ERROR; return code from pthread_create() is %d\n", rc);
        exit(-1);
    }
    if(DUT_join==1){
	assert(my_inputs_DUT.output_0==(PI_0 ^ PI_1));
        pthread_exit(NULL);
    }
}
