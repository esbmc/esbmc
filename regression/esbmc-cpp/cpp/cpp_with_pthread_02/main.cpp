// string assigning
#include <cassert>
#include <pthread.h>

int x;

void *t1(void* arg)
{
  __ESBMC_atomic_begin();
  x++;
  x++;
  __ESBMC_atomic_end();
}

void *t2(void* arg)
{
  __ESBMC_atomic_begin();
  x++;
  x++;
  __ESBMC_atomic_end();
}

int main ()
{
  pthread_t id1, id2;
  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);
  return 0;
}
