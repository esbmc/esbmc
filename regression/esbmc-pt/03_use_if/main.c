#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

struct input_mod1{
    int input_0;
    int input_1;
    int output_0;
};
struct input_mod1 my_input_mod1;

struct input_mod2{
    int input_0;
    int input_1;
    int output_0;
};
struct input_mod2 my_input_mod2;

struct input_DUT{
    int input_0;
    int input_1;
    int output_0;
};
struct input_DUT my_input_DUT;

_Bool join1=0;
_Bool join2=0;
_Bool join3=0;

void *mod1(void *arg)
{
	printf("A mod1 start\n");
	struct input_mod2 *my_struct;
	my_struct = (struct input_mod2 *) arg;
	printf("A inputs: %d %d\n", my_struct->input_0, my_struct->input_1);
	my_struct->output_0 = 1;
	printf("A mod1 output my_struct->output_0: %d\n", my_struct->output_0);
	join1=1;
	pthread_exit(0);
}

void *mod2(void *arg)
{
	printf("B mod2 start\n");
	struct input_mod1 *my_struct;
	my_struct = (struct input_mod1 *) arg;
	printf("B inputs: %d %d\n", my_struct->input_0, my_struct->input_1);
	my_struct->output_0 = 1;
	printf("B mod2 output my_struct->output_0: %d\n", my_struct->output_0);
	join2=1;
	pthread_exit(0);
}

void *DUT(void *arg)
{
   	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
   	void *status;
	int rc;
    	printf("C DUT start\n");
    	struct input_DUT *my_inputs;
    	my_inputs = (struct input_DUT *) arg;
    	my_input_mod2.input_0 = my_inputs->input_0;
    	my_input_mod1.input_0 = my_inputs->input_0;
    	my_input_mod2.input_1 = my_inputs->input_1;

    	pthread_t id0;
    	int t0=0;
    	rc = pthread_create(&id0, &attr, mod1, &my_input_mod2);
	if (rc) {
         	printf("ERROR; return code from pthread_create() is %d\n", rc);
         	exit(-1);
        }
		sleep(1);
//		if (join1==0)
//  	 	  assert(0);
	if(join1==1){
	    	my_input_mod1.input_1 = my_input_mod2.output_0;
	    	pthread_t id1;
	    	int t1=1;
	    	rc = pthread_create(&id1, &attr, mod2, &my_input_mod1);
	    	if (rc) {
		 	printf("ERROR; return code from pthread_create() is %d\n", rc);
		 	exit(-1);
		}	
		sleep(1);
//		if (join2==0)
//  	 	  assert(0);

		if(join2==1){
		    	my_inputs->output_0 = my_input_mod1.output_0;
		    	printf("D DUT output_0: %d \n", my_inputs->output_0);
		    	join3=1;
		    	pthread_exit(NULL);
		}
	}
}
/**********************main C program***********************/
int main(){
    	printf("top start\n");
    	int rc;
    	int PI_0;
    	my_input_DUT.input_0 = PI_0;
    	int PI_1;
    	my_input_DUT.input_1 = PI_1;
	
    	pthread_t id0;
    	rc = pthread_create(&id0, NULL, DUT, &my_input_DUT);
    	if (rc) {
    	        printf("ERROR; return code from pthread_create() is %d\n", rc);
    	        exit(-1);
    	}
		sleep(3);
//		if (join3==0)
//  	 	  assert(0);
	if(join3==1){
		printf("Top output_0: %d\n", my_input_DUT.output_0);
		assert(my_input_DUT.output_0==2);
	}
    	pthread_exit(NULL);
}
