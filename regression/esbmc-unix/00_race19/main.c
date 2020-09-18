#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <stdio.h>

pthread_t id[2];
volatile int results[2];
volatile int done[2]={0,0};
void * one(void * arg) {
   results[0] = 42;
   done[0]=1;
   pthread_exit(NULL);
   }

void * two(void * arg) {
   results[0] = 43;
   done[1]=1;
   pthread_exit(NULL);
   }


int main() {
   pthread_create(&id[0], NULL, one, NULL);
   pthread_create(&id[1], NULL, two, NULL);
   pthread_join(id[0],NULL);
   pthread_join(id[1],NULL);
   if (done[0]&done[1])
     assert (results[0]==42);
   printf("Results are %d and %d.\n",results[0],results[1]);
   exit(EXIT_SUCCESS);
}

