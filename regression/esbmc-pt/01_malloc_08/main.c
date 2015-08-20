#include <stdio.h> 
#include <pthread.h>

void *malloc(unsigned size);
void free(void *p);

#define  N  1

int  num;
pthread_mutex_t *m;
pthread_cond_t *empty, *full;

void * thread1(void * arg)
{
  pthread_mutex_lock(&m);

  while (num > 0) 
    pthread_cond_wait(&empty, &m);
  
  num++;

  pthread_mutex_unlock(&m);
  pthread_cond_signal(&full);
}


void * thread2(void * arg)
{
#if 1
  pthread_mutex_lock(&m);

  while (num == 0) 
    pthread_cond_wait(&full, &m);

//this should cause a deadlock  
//  num--;
  printf("consume ....\n");

  pthread_mutex_unlock(&m);

  pthread_cond_signal(&empty);
#endif
}


int main()
{
  pthread_t  t1, t2;

  num = 1;

  m = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
  empty = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));
  full = (pthread_cond_t *) malloc(sizeof(pthread_cond_t));

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&empty, 0);
  pthread_cond_init(&full, 0);
  
  pthread_create(&t1, 0, thread1, 0);

  pthread_create(&t2, 0, thread2, 0);

  
  pthread_join(t1, 0);
  pthread_join(t2, 0);

  free(m);
  free(empty);
  free(full);

  return 0;  
}
