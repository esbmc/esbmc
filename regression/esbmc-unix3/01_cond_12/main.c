#include <pthread.h>

int  x, y;

pthread_mutex_t m;
pthread_cond_t c, d;

void __ESBMC_yield();

void *thread1(void * arg)
{
  pthread_mutex_lock(&m);
  while (x==0)
  {
    pthread_cond_wait(&c, &m);
  }
  pthread_mutex_unlock(&m);
}

void *thread2(void * arg)
{
  pthread_mutex_lock(&m);
  while (y==0)
  {
    pthread_cond_wait(&d, &m);
  }
  pthread_mutex_unlock(&m);
}

void *thread3(void * arg)
{
  pthread_mutex_lock(&m);
  x=1;
  y=1;
  pthread_mutex_unlock(&m);
//  pthread_cond_signal(&c);
//  pthread_cond_signal(&d);    
}


int main()
{
  pthread_t t1, t2, t3;

  x=0;
  y=0;

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);
  pthread_cond_init(&d, 0);
  
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);
  pthread_create(&t3, 0, thread3, 0);

  return 0;  
}
