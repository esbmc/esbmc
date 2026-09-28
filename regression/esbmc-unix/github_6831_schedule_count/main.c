#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define N 2

int  num;
pthread_mutex_t *m;
pthread_cond_t *empty, *full;

void *thread1(void * arg)
{
  int i;

  i = 0;
  while (i < N){
    pthread_mutex_lock(m);
    while (num > 0) 
      pthread_cond_wait(empty, m);
    
    num++;

    printf ("produce ....%d\n", i);
    pthread_mutex_unlock(m);

    pthread_cond_signal(full);

    i++;
  }
}


void *thread2(void *arg)
{
  int j;

  j = 0;
  while (j < N){
    pthread_mutex_lock(m);
    while (num == 0) 
      pthread_cond_wait(full, m);

    num--;
    printf("consume ....%d\n",j);
    pthread_mutex_unlock(m);
    
    pthread_cond_signal(empty);
    j++;    
  }
}


int main()
{
  pthread_t  t1, t2;

  num = 0;

  m = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
  empty = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));
  full = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));

  if (m==NULL || empty==NULL || full==NULL)
    exit(0);

  pthread_mutex_init(m, 0);
  pthread_cond_init(empty, 0);
  pthread_cond_init(full, 0);
  
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);

  pthread_join(t1, 0);
  pthread_join(t2, 0);

  free(m);
  free(empty);
  free(full);

  return 0;
  
}
