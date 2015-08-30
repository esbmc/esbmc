#include <pthread.h>

int g,x;
pthread_mutex_t mutex=PTHREAD_MUTEX_INITIALIZER;

int nondet_int();

void *t1(void *arg)
{

  pthread_mutex_lock(&mutex);
  if (x>2)
    g=1;              
  else
    g=0;        
  pthread_mutex_unlock(&mutex);
}

void *t2(void *arg)
{
  pthread_mutex_lock(&mutex);
  // this holds due to the lock
  assert(g==0);              
  pthread_mutex_unlock(&mutex);
}

void *t3(void *arg)
{
  pthread_mutex_lock(&mutex);
  assert(g==0);              
  pthread_mutex_unlock(&mutex);
}

int main()
{
  pthread_t id1, id2, id3;

  x=nondet_int();

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  pthread_create(&id3, NULL, t3, NULL);
}
