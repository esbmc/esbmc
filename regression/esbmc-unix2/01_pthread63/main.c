#include <pthread.h>
#define N 2

int a[N], i, j;

void *t1(void *arg)
{
  i = 0;
  a[i] = 2;
}

void *t2(void *arg)
{
  j = 1;
  a[j] = 3;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  pthread_join(id1, NULL);
  pthread_join(id2, NULL);

  assert(a[i] == 2 && a[j] == 3);
  return 0;
}
