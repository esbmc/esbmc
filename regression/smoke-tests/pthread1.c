#include <pthread.h>
#define N 100

int a[N];
unsigned int i, j, nondet_int();

void *t1(void *arg)
{
  i = nondet_int()%N;
  a[i] = *((int *)arg);
}

void *t2(void *arg)
{
  j = nondet_int()%N;
  a[j]=*((int *)arg);
}

int main()
{
  pthread_t id1, id2;
  
  int arg1=10, arg2=20;

  pthread_create(&id1, NULL, t1, &arg1);
  pthread_create(&id2, NULL, t2, &arg2);

  // this should fail
  assert(a[i]==10 && a[j]==20);

  return 0;
}
