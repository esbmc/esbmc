#include <pthread.h>
#include <stdio.h>

#define QUEUE_FULL_SIZE 1

// global variables
pthread_mutex_t mux;
pthread_cond_t  cond_full;
pthread_cond_t  cond_empty;

int qsize;

int static counter = 0;

// producer thread
void* producer(void* argv)
{
  int i;
   for (i = 0; i < 3; i ++) {
//    while (1) {
 
    pthread_mutex_lock(&mux);
    
    while (qsize == QUEUE_FULL_SIZE)
      pthread_cond_wait(&cond_full, &mux);

    //printf("  produce %d, item, qsize = %d\n", counter , qsize);
    counter ++;
    pthread_cond_signal(&cond_empty);
    
   // sleep(1);
    qsize ++;    
    pthread_mutex_unlock(&mux);
  }
}


// consumer thread
void* consumer(void* argv)
{
  int val, i;

  for (i = 0; i < 3; i ++) {
  //  while (1) {
    pthread_mutex_lock(&mux);
    
    if (qsize == 0)
      pthread_cond_wait(&cond_empty, &mux);

//    printf("consume %d,  item, qsize = %d \n",val, qsize);
    pthread_cond_signal(&cond_full);
    
    qsize --;
    pthread_mutex_unlock(&mux);
  }
}


// main procedure
int main()
{
  int i, ret1, ret2;

  pthread_t prod[2];
  pthread_t cons[2];

  qsize = 0;

  pthread_mutex_init(&mux, NULL);
  pthread_cond_init(&cond_full, NULL);
  pthread_cond_init(&cond_empty, NULL);

  for (i = 0; i < 2; i ++) {
    /*ret1 =*/ pthread_create(&prod[i], NULL, producer, NULL);
    /*ret2 =*/ pthread_create(&cons[i], NULL, consumer, NULL);

    if (ret1 != 0 || ret2 != 0) {
      //cout << "Error creating threads." << endl;
      exit(-1);
    }
  }
  //  pthread_exit(NULL);
  for (i = 0; i < 2; i ++) {
    pthread_join(prod[i], NULL);
    pthread_join(cons[i], NULL);
  }

  pthread_mutex_destroy(&mux);
  pthread_cond_destroy(&cond_full);
  pthread_cond_destroy(&cond_empty);

}
