#include <pthread.h>

//#define NULL 0

//typedef int pthread_t;

//int pthread_create(pthread_t *, void *, void *, void *);

int g;

void *t1(void *arg)
{
  int a;
  
  a=*((int *)arg);

  assert(a==10);
}

void *t2(void *arg)
{
  int a;
  
  a=*((int *)arg);

  assert(a==20);

  g=1;
}

int main()
{
  pthread_t id1, id2;
  
  int arg1=10, arg2=20;

  pthread_create(&id1, NULL, t1, &arg1);
  pthread_create(&id2, NULL, t2, &arg2);
  
  // this should fail
  assert(g==0);
}
