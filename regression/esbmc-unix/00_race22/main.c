#include <pthread.h>
#include <assert.h>

int nondet_int();
int x=0;

void *t1(void *arg)
{
  assert(x==0 || x==1 || x==2);
  return NULL;
}

void *t2(void *arg)
{
  int y=nondet_int();
  if (y==0 || y==1 || y==2) x=y;
  return NULL;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  return 0;
}
