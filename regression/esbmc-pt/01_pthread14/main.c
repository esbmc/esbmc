#include <pthread.h>
#define N 10

int a[N], i, j=1, x=2;
int nondet_int();

void *t1(void *arg)
{
  if (x>2)
  {
    a[i] = ((int *)arg);
	//this should fail
    assert(i>=0 && i<N);
  }
}

void *t2(void *arg)
{
  if (x>3)
    a[j]=*((int *)arg);
  else
    x=3;
}

int main()
{
  pthread_t id1, id2;
  
  int arg1=10, arg2=20;
  i = -1;//nondet_int();
  pthread_create(&id1, NULL, t1, &arg1);
  pthread_create(&id2, NULL, t2, &arg2);  
}
