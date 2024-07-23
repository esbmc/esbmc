#include <pthread.h>
#include <assert.h>

int x=0;

void *t1(void *arg)
{
  assert(x==0);
}

void *t2(void *arg)
{
  int y;
  if (y==1) x=y;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  return 0;
}
