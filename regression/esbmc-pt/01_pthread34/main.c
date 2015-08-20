#include <pthread.h>
#include <assert.h>

void *t1(void *arg)
{
  int x;
  x=0;
  assert(x==0);
}

void *t2(void *arg)
{
  int y;
  y=0;
  assert(y==0);
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  return 0;
}

