// #10293: q may point to g, which main writes without a lock.
#include <pthread.h>
#include <stdlib.h>

_Bool nondet_bool();

struct S
{
  int *p;
};

int g;

void *t1(void *arg)
{
  struct S *s = malloc(sizeof(struct S));
  __ESBMC_assume(s);
  int *q = s->p;
  if (nondet_bool())
    q = &g;
  if (q)
    *q = *q + 1;
  return 0;
}

int main()
{
  pthread_t id;
  pthread_create(&id, 0, t1, 0);
  g = 1;
  return 0;
}
