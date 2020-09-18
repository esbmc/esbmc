#include <pthread.h>

int  x;

pthread_mutex_t m;
pthread_cond_t c;

void *thread1(void * arg)
{
  pthread_mutex_lock(&m);
  while (x==0)
  {
    pthread_cond_wait(&c, &m);
  }

  pthread_mutex_unlock(&m);
  pthread_cond_signal(&c);  
}

void *thread2(void * arg)
{
  int i;

  pthread_mutex_lock(&m);
  pthread_cond_wait(&c, &m);

  x=1;

  pthread_mutex_unlock(&m);
  pthread_cond_signal(&c);  
}


int main()
{
  pthread_t  t1, t2;

  x=1;

  pthread_mutex_init(&m, 0);
  pthread_cond_init(&c, 0);
  
  pthread_create(&t1, 0, thread1, 0);
  pthread_create(&t2, 0, thread2, 0);

  return 0;  
}
