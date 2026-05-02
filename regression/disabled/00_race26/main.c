#include <pthread.h>
#include <assert.h>

int nondet_int();
int x=0;

void *t1(void *arg)
{
  assert((x^1)==0 || (x^2)==1 || (x^3)==2);
  return NULL;
}

void *t2(void *arg)
{
  int y=nondet_int();
  if ((y^1)==0 || (y^2)==1 || (y^3)==2) x=y;
  return NULL;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

  pthread_join(id1, NULL);
  pthread_join(id2, NULL);

  return 0;
}
