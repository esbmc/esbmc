#include <pthread.h>
#define N 3

int a[N], i, j;
//unsigned int nondet_int();
#if 0
int funcA(void)
{
  return 0;//nondet_int()%3;
}

int funcB(void)
{
  return 1;//nondet_int()%2;
}
#endif

void *t1(void *arg)
{
  i = 0; //funcA();
  a[i] = 2;//*((int *)arg);
}

void *t2(void *arg)
{
  j = 1; //funcB();
  a[j]=3;//*((int *)arg);

}

int main()
{
  pthread_t id1, id2;
  
  int arg1=10, arg2=20;

  pthread_create(&id1, NULL, t1, NULL/*&arg1*/);
  pthread_create(&id2, NULL, t2, NULL/*&arg2*/);
  
  // this should fail
  assert(a[i]==10 && a[j]==20);
}
