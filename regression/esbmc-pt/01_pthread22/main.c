#include <pthread.h>

int g;
void __ESBMC_yield();

void *t1(void *arg)
{
  g=1;
}

void *t2(void *arg)
{
  g=2;
}

int main()
{
  pthread_t id1, id2;

  pthread_create(&id1, NULL, t1, NULL);
  pthread_create(&id2, NULL, t2, NULL);

// __ESBMC_yield();

  assert(g==0 || g==2);
}
