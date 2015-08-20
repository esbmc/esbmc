#include <pthread.h>
#define N 10

int a[N], i=1, j=2;

void *t1(void *arg)
{
  a[i] = 1;
}

void *t2(void *arg)
{
  a[j] = 5;
}

int main()
{
  pthread_t id1, id2;
  __ESBMC_assume(i>=0 && i<N);
  __ESBMC_assume(j>=0 && j<N);
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);  

  assert(a[i]==1 && a[j]==5);
}
